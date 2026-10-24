import { create, fromBinary, toBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema, NodeProtoSchema, TensorProtoSchema, ValueInfoProtoSchema, type NodeProto } from 'onnx-buf';
import type { HeadWeights, ModelManifest } from './types';

function fp32(name: string, dims: number[], values: Float32Array) {
  const rawData = new Uint8Array(values.length * 4);
  const view = new DataView(rawData.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return create(TensorProtoSchema, { name, dims: dims.map(BigInt), dataType: 1, rawData });
}

/** Export this prepared model recipe only. This is not a generic ONNX fine-tuning converter. */
export function exportClassifier(bytes: Uint8Array, manifest: ModelManifest, head: HeadWeights, labels: string[]): Uint8Array {
  if (manifest.schemaVersion !== 1 || !Number.isInteger(head.classes) || head.classes < 2 || head.classes > 32 ||
      !Number.isInteger(head.featureSize) || head.featureSize < 1 || head.featureSize !== manifest.features.size ||
      head.weights.length !== head.featureSize * head.classes || head.bias.length !== head.classes ||
      !head.weights.every(Number.isFinite) || !head.bias.every(Number.isFinite)) {
    throw new Error('Invalid trained head or model recipe');
  }
  const names = labels.map(s => s.trim().normalize('NFC'));
  if (names.length !== head.classes || new Set(names).size !== names.length || names.some(s => s.length < 1 || s.length > 80)) {
    throw new Error('Labels must be unique, nonempty and match the trained class count');
  }
  const model = fromBinary(ModelProtoSchema, bytes);
  const graph = model.graph;
  if (!graph || model.functions.length || graph.sparseInitializer.length || graph.input.length !== 1 || graph.input[0].name !== manifest.input.name) {
    throw new Error('Unsupported graph or required inputs');
  }
  if (graph.initializer.some(t => t.externalData.length || t.dataLocation === 1)) throw new Error('External model weights are not supported');
  if (graph.node.some(n => n.attribute.some(a => a.g || a.graphs.length))) throw new Error('Nested graphs are not supported by this recipe');
  const original = graph.node.find(n => n.name === manifest.head.nodeName);
  if (!original || original.opType !== 'Gemm' || original.input[0] !== manifest.features.name ||
      original.input[1] !== manifest.head.weightName || original.input[2] !== manifest.head.biasName ||
      !original.output.includes(manifest.originalOutput)) throw new Error('Original classifier mapping does not match the prepared graph');
  const featureInfo = graph.output.find(v => v.name === manifest.features.name);
  const featureType = featureInfo?.type?.value;
  if (featureType?.case !== 'tensorType' || featureType.value.elemType !== 1 ||
      featureType.value.shape?.dim[1]?.value.case !== 'dimValue' ||
      Number(featureType.value.shape.dim[1].value.value) !== head.featureSize) throw new Error('Feature shape does not match the trained head');

  const producers = new Map<string, NodeProto>();
  for (const node of graph.node) for (const output of node.output) {
    if (producers.has(output)) throw new Error('Duplicate tensor producer');
    producers.set(output, node);
  }
  const initializers = new Set(graph.initializer.map(t => t.name));
  const inputs = new Set(graph.input.map(t => t.name));
  const neededTensors = new Set<string>();
  const neededNodes = new Set<NodeProto>();
  const visiting = new Set<string>();
  const visit = (name: string) => {
    if (!name || neededTensors.has(name)) return;
    if (visiting.has(name)) throw new Error('Cyclic graph');
    visiting.add(name);
    if (!initializers.has(name) && !inputs.has(name)) {
      const node = producers.get(name);
      if (!node) throw new Error(`Missing frozen tensor: ${name}`);
      node.input.forEach(visit);
      neededNodes.add(node);
    }
    visiting.delete(name);
    neededTensors.add(name);
  };
  visit(manifest.features.name);
  const reserved = ['ovb.head.weight', 'ovb.head.bias', 'ovb.head.matmul'];
  if (reserved.some(name => neededTensors.has(name))) throw new Error('Reserved output name already exists');
  graph.node = graph.node.filter(node => neededNodes.has(node));
  graph.initializer = graph.initializer.filter(t => neededTensors.has(t.name!));
  graph.valueInfo = graph.valueInfo.filter(v => neededTensors.has(v.name!));
  graph.initializer.push(fp32(reserved[0], [head.featureSize, head.classes], head.weights), fp32(reserved[1], [head.classes], head.bias));
  graph.node.push(
    create(NodeProtoSchema, { name: 'ovb.head.linear', opType: 'MatMul', input: [manifest.features.name, reserved[0]], output: [reserved[2]] }),
    create(NodeProtoSchema, { name: 'ovb.head.add', opType: 'Add', input: [reserved[2], reserved[1]], output: ['logits'] }),
  );
  graph.output = [create(ValueInfoProtoSchema, { name: 'logits', type: { value: { case: 'tensorType', value: { elemType: 1, shape: { dim: [1, head.classes].map(d => ({ value: { case: 'dimValue' as const, value: BigInt(d) } })) } } } } })];
  model.trainingInfo = [];
  model.metadataProps = model.metadataProps.filter(p => !p.key?.startsWith('ovb.'));
  model.metadataProps.push({ $typeName: 'onnx.StringStringEntryProto', key: 'ovb.labels', value: JSON.stringify(names) }, { $typeName: 'onnx.StringStringEntryProto', key: 'ovb.recipe', value: manifest.id });
  return toBinary(ModelProtoSchema, model);
}
