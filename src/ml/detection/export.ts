import { create, fromBinary, toBinary } from '@bufbuild/protobuf';
import { ModelProtoSchema, TensorProtoSchema, type TensorProto, type ValueInfoProto } from 'onnx-buf';

export interface DFineExportRecipe {
  schemaVersion: 1;
  id: 'dfine-nano-coco-final-head-v1';
  input: { name: string; shape: number[] };
  outputs: { name: string; shape: number[] }[];
  parameters: { parameter: string; initializers: string[]; shape: number[] }[];
}

interface UpdatedTensor { shape: number[]; data: Float32Array }

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((n, i) => n === right[i]);
}

function elementCount(shape: readonly number[]): number {
  if (!shape.length || shape.some(n => !Number.isSafeInteger(n) || n < 1)) throw new Error('Invalid parameter shape');
  const count = shape.reduce((total, n) => total * n, 1);
  if (!Number.isSafeInteger(count)) throw new Error('Parameter shape is too large');
  return count;
}

function matchesInfo(info: ValueInfoProto, expected: { name: string; shape: number[] }): boolean {
  const type = info.type?.value;
  if (info.name !== expected.name || type?.case !== 'tensorType' || type.value.elemType !== 1) return false;
  const dims = type.value.shape?.dim;
  return Boolean(dims && dims.length === expected.shape.length && dims.every((d, i) =>
    d.value.case === 'dimValue' && Number(d.value.value) === expected.shape[i]));
}

/** Replace only parameters mapped by the checksummed, prepared D-FINE recipe.
 * The caller verifies the source artifact digest before passing its bytes here.
 * All graph nodes and all unmapped weights are retained in the standalone model.
 */
export function exportDFineOnnx(
  source: Uint8Array,
  recipe: DFineExportRecipe,
  updates: Record<string, UpdatedTensor>,
): Uint8Array {
  if (recipe.schemaVersion !== 1 || recipe.id !== 'dfine-nano-coco-final-head-v1' || !recipe.parameters.length) {
    throw new Error('Unsupported detector export recipe');
  }
  const expectedKeys = recipe.parameters.map(p => p.parameter);
  if (expectedKeys.some(k => !k) || new Set(expectedKeys).size !== expectedKeys.length ||
      Object.keys(updates).length !== expectedKeys.length || expectedKeys.some(k => !Object.hasOwn(updates, k))) {
    throw new Error('Trained parameters must match the complete recipe parameter set');
  }
  const model = fromBinary(ModelProtoSchema, source);
  const graph = model.graph;
  if (!graph || model.functions.length || graph.sparseInitializer.length) throw new Error('Unsupported complete detector graph');
  if (graph.initializer.some(t => t.externalData.length || t.dataLocation === 1) ||
      graph.node.some(n => n.attribute.some(a => a.t?.externalData.length || a.t?.dataLocation === 1 ||
        a.tensors.some(t => t.externalData.length || t.dataLocation === 1)))) {
    throw new Error('External model weights cannot be exported as a complete model');
  }
  if (graph.node.some(n => n.attribute.some(a => a.g || a.graphs.length))) throw new Error('Nested detector graphs are unsupported');
  if (graph.input.length !== 1 || !matchesInfo(graph.input[0], recipe.input)) throw new Error('Detector input does not match the recipe');
  if (recipe.outputs.length !== 2 || new Set(recipe.outputs.map(o => o.name)).size !== 2) throw new Error('Expected distinct logits and boxes outputs');
  for (const output of recipe.outputs) {
    const info = graph.output.find(o => o.name === output.name);
    if (!info || !matchesInfo(info, output)) throw new Error(`Detector output does not match the recipe: ${output.name}`);
  }

  const initializers = new Map(graph.initializer.map(t => [t.name, t]));
  if (initializers.size !== graph.initializer.length) throw new Error('Duplicate initializer names');
  const replacements = new Map<string, TensorProto>();
  for (const binding of recipe.parameters) {
    const count = elementCount(binding.shape);
    const update = updates[binding.parameter];
    if (!(update.data instanceof Float32Array) || !sameShape(update.shape, binding.shape) || update.data.length !== count) {
      throw new Error(`Trained parameter shape does not match: ${binding.parameter}`);
    }
    if (!update.data.every(Number.isFinite)) throw new Error(`Trained parameter must be finite: ${binding.parameter}`);
    if (!binding.initializers.length) throw new Error(`Missing initializer mapping: ${binding.parameter}`);
    const rawData = new Uint8Array(count * 4);
    const view = new DataView(rawData.buffer);
    for (let i = 0; i < count; i++) view.setFloat32(i * 4, update.data[i], true);
    for (const name of binding.initializers) {
      const original = initializers.get(name);
      if (!name || replacements.has(name)) throw new Error(`Conflicting initializer mapping: ${name}`);
      if (!original || original.dataType !== 1 || !sameShape(original.dims.map(Number), binding.shape)) {
        throw new Error(`Source initializer does not match the trained parameter: ${name}`);
      }
      replacements.set(name, create(TensorProtoSchema, {
        name, dims: original.dims, dataType: 1, rawData: rawData.slice(), docString: original.docString,
      }));
    }
  }
  graph.initializer = graph.initializer.map(t => replacements.get(t.name) ?? t);
  graph.output = recipe.outputs.map(expected => graph.output.find(o => o.name === expected.name)!);
  model.trainingInfo = [];
  model.metadataProps = model.metadataProps.filter(p => !p.key.startsWith('ovb.'));
  model.metadataProps.push(
    { $typeName: 'onnx.StringStringEntryProto', key: 'ovb.recipe', value: recipe.id },
    { $typeName: 'onnx.StringStringEntryProto', key: 'ovb.trainedParameters', value: JSON.stringify(expectedKeys) },
  );
  return toBinary(ModelProtoSchema, model);
}
