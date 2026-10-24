import { useEffect, useRef, useState } from 'react';
import type { Sample, Split, WorkerRequest, WorkerResponse } from './protocol';

const MAX_IMAGES = 64;
const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 16_000_000;
const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
type PreviewSample = Sample & { preview: string; synthetic: boolean };
type Result = Extract<WorkerResponse, { type: 'result' }>;
type Progress = Extract<WorkerResponse, { type: 'progress' }>;
type BuildInfo = { commit?: string; sha?: string; pr?: number | string };

function Icon({ kind }: { kind: 'plus' | 'arrow' | 'download' | 'check' | 'shield' | 'image' }) {
  const paths = {
    plus: <path d="M12 5v14M5 12h14" />,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    download: <path d="M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4" />,
    check: <path d="m5 12 4 4L19 6" />,
    shield: <><path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z" /><path d="m8 12 3 3 5-6" /></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8" cy="8" r="1.5" /><path d="m4 17 5-5 4 4 3-3 5 5" /></>,
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[kind]}</svg>;
}

function syntheticImage(label: number, index: number): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const context = canvas.getContext('2d');
  if (!context) return Promise.reject(new Error('サンプル画像を作成できませんでした。'));
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, 224, 224);
  const colors = label === 0
    ? ['#e85331', '#ec7829', '#d84240', '#ed6635', '#dc5540', '#e98633']
    : ['#2764c7', '#257fae', '#3b58bc', '#2671cb', '#315eb9', '#3287b2'];
  context.fillStyle = colors[index];
  const x = 106 + ((index * 13) % 29) - 14;
  const y = 110 + ((index * 17) % 23) - 11;
  const size = 62 + (index % 3) * 5;
  if (index % 2 === 0) {
    context.beginPath();
    context.arc(x, y, size, 0, Math.PI * 2);
    context.fill();
  } else {
    context.fillRect(x - size, y - size, size * 2, size * 2);
  }
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob);
    else reject(new Error('サンプル画像を作成できませんでした。'));
  }, 'image/png'));
}

function formatDuration(ms: number) {
  const seconds = ms / 1000;
  return seconds < 60 ? `${seconds.toFixed(1)}秒` : `${Math.floor(seconds / 60)}分${Math.round(seconds % 60)}秒`;
}

export default function App() {
  const [labels, setLabels] = useState(['クラス1', 'クラス2']);
  const [samples, setSamples] = useState<PreviewSample[]>([]);
  const [epochs, setEpochs] = useState('40');
  const [learningRate, setLearningRate] = useState('0.01');
  const [busy, setBusy] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [buildInfo, setBuildInfo] = useState<BuildInfo | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const jobIdRef = useRef(0);
  const samplesRef = useRef<PreviewSample[]>([]);
  const urlsRef = useRef(new Set<string>());
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}build-info.json`, { signal: controller.signal })
      .then(response => response.ok ? response.json() : null)
      .then((value: unknown) => {
        if (value && typeof value === 'object' && mountedRef.current) {
          const info = value as Record<string, unknown>;
          setBuildInfo({
            commit: typeof info.commit === 'string' ? info.commit : undefined,
            sha: typeof info.sha === 'string' ? info.sha : undefined,
            pr: typeof info.pullRequest === 'number' || typeof info.pullRequest === 'string' ? info.pullRequest : undefined,
          });
        }
      }).catch(() => { /* Build metadata is optional in local development. */ });
    return () => {
      mountedRef.current = false;
      controller.abort();
      workerRef.current?.terminate();
      workerRef.current = null;
      jobIdRef.current += 1;
      for (const url of urlsRef.current) URL.revokeObjectURL(url);
      urlsRef.current.clear();
    };
  }, []);

  const disabled = busy || preparing;
  const trimmedLabels = labels.map(label => label.trim().normalize('NFC'));
  const labelError = trimmedLabels.some(label => !label)
    ? 'すべてのクラスに名前を付けてください。空白だけの名前は使えません。'
    : trimmedLabels.some(label => label.length > 80)
      ? 'クラス名は80文字以内にしてください。'
      : new Set(trimmedLabels).size !== labels.length
      ? 'クラス名が重複しています。それぞれ異なる名前を付けてください。'
      : '';
  const counts = labels.map((_, label) => ({
    train: samples.filter(sample => sample.label === label && sample.split === 'train').length,
    test: samples.filter(sample => sample.label === label && sample.split === 'test').length,
  }));
  const trainingCount = counts.reduce((sum, count) => sum + count.train, 0);
  const testCount = counts.reduce((sum, count) => sum + count.test, 0);
  const dataReady = counts.every(count => count.train > 0 && count.test > 0);
  const epochCount = Number(epochs);
  const rate = Number(learningRate);
  const parametersValid = Number.isInteger(epochCount) && epochCount >= 1 && epochCount <= 200
    && Number.isFinite(rate) && rate >= 0.001 && rate <= 0.1;
  const canRun = !disabled && !labelError && dataReady && parametersValid;
  const hasSyntheticImages = samples.some(sample => sample.synthetic);
  const fraction = Math.max(0, Math.min(1, progress?.fraction ?? 0));

  function invalidate() {
    workerRef.current?.terminate();
    workerRef.current = null;
    jobIdRef.current += 1;
    setResult(null);
    setProgress(null);
    setError('');
    setNotice('');
  }

  function previewSample(sample: Sample, synthetic: boolean): PreviewSample {
    const preview = URL.createObjectURL(sample.file);
    urlsRef.current.add(preview);
    return { ...sample, preview, synthetic };
  }

  function updateSamples(next: PreviewSample[]) {
    const retained = new Set(next.map(sample => sample.preview));
    for (const sample of samplesRef.current) {
      if (!retained.has(sample.preview)) {
        URL.revokeObjectURL(sample.preview);
        urlsRef.current.delete(sample.preview);
      }
    }
    samplesRef.current = next;
    setSamples(next);
  }

  async function useSample() {
    if (disabled) return;
    invalidate();
    setPreparing(true);
    try {
      const generated = await Promise.all([0, 1].flatMap(label =>
        Array.from({ length: 6 }, async (_, index): Promise<Sample> => ({
          id: `demo-${label}-${index}`,
          name: `${label === 0 ? 'warm' : 'cool'}-${index + 1}.png`,
          label,
          split: index < 4 ? 'train' : 'test',
          file: await syntheticImage(label, index),
        }))));
      if (!mountedRef.current) return;
      setLabels(['暖色', '寒色']);
      updateSamples(generated.map(sample => previewSample(sample, true)));
      setNotice('サンプルを用意しました。「学習してONNXを作成」から試せます。');
    } catch (caught) {
      if (mountedRef.current) setError(caught instanceof Error ? caught.message : 'サンプルを用意できませんでした。');
    } finally {
      if (mountedRef.current) setPreparing(false);
    }
  }

  async function importImages(files: File[], label: number, split: Split) {
    if (disabled || !files.length) return;
    invalidate();
    setPreparing(true);
    try {
      if (samplesRef.current.length + files.length > MAX_IMAGES) throw new Error('画像は合計64枚までです。追加する枚数を減らしてください。');
      for (const file of files) {
        if (!IMAGE_TYPES.has(file.type)) throw new Error(`「${file.name}」は対応していない形式です。PNG・JPEG・WebPを選んでください。`);
        if (file.size > MAX_FILE_BYTES) throw new Error(`「${file.name}」は20MiBを超えています。画像を小さくしてください。`);
        let bitmap: ImageBitmap;
        try { bitmap = await createImageBitmap(file); }
        catch { throw new Error(`「${file.name}」を画像として読み込めませんでした。別のファイルを選んでください。`); }
        const pixels = bitmap.width * bitmap.height;
        bitmap.close();
        if (pixels > MAX_IMAGE_PIXELS) throw new Error(`「${file.name}」は16メガピクセルを超えています。画像を縮小してください。`);
      }
      if (!mountedRef.current) return;
      const additions = files.map(file => previewSample({
        id: crypto.randomUUID(), name: file.name, label, split, file,
      }, false));
      updateSamples([...samplesRef.current, ...additions]);
    } catch (caught) {
      if (mountedRef.current) setError(caught instanceof Error ? caught.message : '画像を読み込めませんでした。');
    } finally {
      if (mountedRef.current) setPreparing(false);
    }
  }

  function removeClass(label: number) {
    if (disabled || labels.length <= 2) return;
    invalidate();
    setLabels(labels.filter((_, index) => index !== label));
    updateSamples(samplesRef.current.filter(sample => sample.label !== label)
      .map(sample => sample.label > label ? { ...sample, label: sample.label - 1 } : sample));
  }

  function cancelRun() {
    workerRef.current?.terminate();
    workerRef.current = null;
    jobIdRef.current += 1;
    setBusy(false);
    setResult(null);
    setProgress(null);
    setError('');
    setNotice('学習を中断しました。');
  }

  function startRun() {
    if (!canRun) return;
    invalidate();
    const id = ++jobIdRef.current;
    setBusy(true);
    setProgress({ type: 'progress', id, stage: '学習の準備をしています', fraction: 0 });
    try {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      const isCurrent = () => jobIdRef.current === id && workerRef.current === worker;
      const finish = () => {
        worker.terminate();
        workerRef.current = null;
        setBusy(false);
      };
      worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
        if (!isCurrent() || data.id !== id) return;
        if (data.type === 'progress') setProgress(data);
        else if (data.type === 'result') {
          setResult(data);
          setProgress(null);
          finish();
        } else if (data.type === 'cancelled') {
          setResult(null);
          setProgress(null);
          setNotice('学習を中断しました。');
          finish();
        } else if (data.type === 'error') {
          setResult(null);
          setProgress(null);
          setError(data.message);
          finish();
        }
      };
      worker.onerror = event => {
        event.preventDefault();
        if (!isCurrent()) return;
        setResult(null);
        setProgress(null);
        setError(`学習を完了できませんでした。${event.message || '画像や設定を確認して、もう一度お試しください。'}`);
        finish();
      };
      worker.onmessageerror = () => {
        if (!isCurrent()) return;
        setResult(null);
        setProgress(null);
        setError('学習結果を読み取れませんでした。もう一度お試しください。');
        finish();
      };
      const request: WorkerRequest = {
        type: 'run', id, labels: trimmedLabels,
        samples: samples.map(({ id: sampleId, name, label, split, file }) => ({ id: sampleId, name, label, split, file })),
        epochs: epochCount, learningRate: rate,
      };
      worker.postMessage(request);
    } catch (caught) {
      workerRef.current?.terminate();
      workerRef.current = null;
      setBusy(false);
      setProgress(null);
      setError(caught instanceof Error ? caught.message : 'このブラウザで学習を開始できませんでした。');
    }
  }

  function downloadModel() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([result.archive], { type: 'application/zip' }));
    urlsRef.current.add(url);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'open-vision-model.zip';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => {
      URL.revokeObjectURL(url);
      urlsRef.current.delete(url);
    }, 10_000);
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#" aria-label="Open Vision Builder ホーム">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>
          <span>Open Vision <strong>Builder</strong></span>
        </a>
        <span className="local-badge"><span />ローカル実行</span>
      </header>
      <main>
        <section className="intro" aria-labelledby="page-heading">
          <div className="intro-copy"><p className="eyebrow">YOUR IMAGES. YOUR MODEL.</p><h1 id="page-heading">画像から、自分の分類モデルを。</h1><p className="intro-lead">画像の準備、学習、ONNXの書き出しを、このブラウザで。</p></div>
          <div className="scope-note"><span className="preview-badge">分類の技術検証版</span><p>MobileNetV4 Conv Small の特徴抽出部分を固定し、分類部分を学習します。物体検出は未対応です。</p></div>
        </section>
        <ol className="workflow-nav" aria-label="作業の流れ">
          <li className={samples.length ? 'is-complete' : 'is-current'}><a href="#dataset"><span>01</span><div><strong>画像を準備</strong><small>クラスと学習・評価データ</small></div></a></li>
          <li className={busy ? 'is-current' : result ? 'is-complete' : ''}><a href="#training"><span>02</span><div><strong>ブラウザで学習</strong><small>分類部分の重みを更新</small></div></a></li>
          <li className={result ? 'is-current' : ''}><a href="#result"><span>03</span><div><strong>モデルを持ち出す</strong><small>検証済みのONNXを保存</small></div></a></li>
        </ol>
        <div className="local-note"><Icon kind="shield" /><p><strong>画像はこの端末で処理します。</strong> アップロードは行いません。この検証版ではデータを自動保存しません。</p></div>

        <section className="panel dataset-panel" id="dataset" aria-labelledby="dataset-heading">
          <div className="section-heading"><div className="section-title"><span className="step-number">01</span><div><h2 id="dataset-heading">画像を準備する</h2><p>クラスごとに、学習用と評価用の画像を分けて追加します。</p></div></div><button type="button" className="button button-secondary sample-button" disabled={disabled} onClick={useSample}><Icon kind="image" />サンプルで試す</button></div>
          <div className="dataset-toolbar"><p>PNG・JPEG・WebP · 1枚20MiB / 16MPまで · 合計64枚まで</p><span className="count-pill">{samples.length} / 64 枚</span></div>
          {labelError && <p className="inline-error" role="alert">{labelError}</p>}
          <div className="class-grid">
            {labels.map((label, classIndex) => (
              <article className="class-card" key={classIndex}>
                <div className="class-header"><span className="class-number" aria-hidden="true">{String(classIndex + 1).padStart(2, '0')}</span><label className="class-name"><span>クラス名</span><input aria-label={`クラス${classIndex + 1}の名前`} value={label} disabled={disabled} maxLength={80} placeholder="例：マグカップ" aria-invalid={!trimmedLabels[classIndex] || trimmedLabels.filter(name => name === trimmedLabels[classIndex]).length > 1} onChange={event => { invalidate(); setLabels(labels.map((name, index) => index === classIndex ? event.target.value : name)); }} /></label>{labels.length > 2 && <button type="button" className="text-button remove-class" aria-label={`クラス${classIndex + 1}を削除`} disabled={disabled} onClick={() => removeClass(classIndex)}>削除</button>}</div>
                {(['train', 'test'] as const).map(split => {
                  const group = samples.filter(sample => sample.label === classIndex && sample.split === split);
                  const title = split === 'train' ? '学習用' : '評価用';
                  return <div className={`image-group ${split}`} key={split}>
                    <div className="group-heading"><h3>{title}<span>{group.length}枚</span></h3><label className={`file-picker${disabled || samples.length >= MAX_IMAGES ? ' is-disabled' : ''}`}><Icon kind="plus" /><span>画像を追加</span><input type="file" accept="image/png,image/jpeg,image/webp" multiple disabled={disabled || samples.length >= MAX_IMAGES} aria-label={`クラス${classIndex + 1}の${title}画像を追加`} onChange={event => { const files = Array.from(event.currentTarget.files ?? []); event.currentTarget.value = ''; void importImages(files, classIndex, split); }} /></label></div>
                    {group.length > 0 ? <ul className="thumbnail-list" aria-label={`${label || `クラス${classIndex + 1}`}の${title}画像`}>{group.map(sample => <li key={sample.id} title={sample.name}><img src={sample.preview} alt={sample.name} loading="lazy" /><button type="button" className="remove-image" disabled={disabled} aria-label={`${sample.name}を削除`} onClick={() => { invalidate(); updateSamples(samplesRef.current.filter(item => item.id !== sample.id)); }}>×</button></li>)}</ul> : <div className="empty-images"><Icon kind="image" /><span>{split === 'train' ? 'モデルに覚えてほしい画像' : '学習に使わない、別の画像'}</span></div>}
                    <p className="group-hint">{split === 'train' ? 'このクラスを学習する画像 · 1枚以上' : '学習後の正解率を確かめる画像 · 1枚以上'}</p>
                  </div>;
                })}
              </article>
            ))}
          </div>
          <div className="dataset-footer"><button type="button" className="text-button add-class" disabled={disabled || labels.length >= 8} onClick={() => { invalidate(); let next = labels.length + 1; while (trimmedLabels.includes(`クラス${next}`)) next += 1; setLabels([...labels, `クラス${next}`]); }}><Icon kind="plus" />クラスを追加<span>（最大8）</span></button><p>評価用には学習用と異なる画像を選んでください。重複画像は学習時に検出します。</p></div>
          {samples.length > 0 && <p className="sample-note">「サンプルで試す」を押すと、現在のクラス・画像・学習結果をサンプルへ置き換えます。</p>}
          {hasSyntheticImages && <p className="sample-note">サンプルは、このブラウザで生成した暖色・寒色の図形12枚です。動作確認用のため、自然画像での精度や実用性能を示すものではありません。</p>}
          {preparing && <p className="status-notice" role="status">画像を準備しています…</p>}
        </section>

        <div className="workflow-bottom">
          <section className="panel training-panel" id="training" aria-labelledby="training-heading">
            <div className="section-title"><span className="step-number">02</span><div><h2 id="training-heading">分類モデルを学習する</h2><p>画像を用意したら、学習を開始できます。</p></div></div>
            <div className="dataset-totals"><div><strong>{labels.length}</strong><span>クラス</span></div><div><strong>{trainingCount}</strong><span>学習画像</span></div><div><strong>{testCount}</strong><span>評価画像</span></div></div>
            <details className="settings"><summary>学習の設定<span>{epochs || '—'}エポック</span></summary><div className="settings-fields"><label>エポック数<input type="number" min="1" max="200" step="1" value={epochs} disabled={disabled} onChange={event => { invalidate(); setEpochs(event.target.value); }} /><small>繰り返し学習する回数 · 1〜200</small></label><label>学習率<input type="number" min="0.001" max="0.1" step="any" value={learningRate} disabled={disabled} onChange={event => { invalidate(); setLearningRate(event.target.value); }} /><small>1回の更新の大きさ · 0.001〜0.1</small></label></div></details>
            {!parametersValid && <p className="inline-error" role="alert">エポック数は1〜200の整数、学習率は0.001〜0.1で指定してください。</p>}
            {!dataReady && <p className="readiness-note">各クラスに学習用・評価用を1枚ずつ以上追加すると、学習を開始できます。</p>}
            {busy && <div className="progress-area" role="status" aria-live="polite"><div className="progress-label"><strong>{progress?.stage}</strong><span>{Math.round(fraction * 100)}%</span></div><progress max="1" value={fraction} aria-label="学習と書き出しの進捗" />{progress?.loss !== undefined && <p>現在の損失 {progress.loss.toFixed(5)}</p>}<p>モデルの読み込みからONNXの検証まで、この端末で処理しています。</p></div>}
            <div className="training-actions"><button type="button" className="button button-primary" disabled={!canRun} onClick={startRun}>学習してONNXを作成<Icon kind="arrow" /></button>{busy && <button type="button" className="button button-cancel" onClick={cancelRun}>中断</button>}</div>
            <p className="training-footnote">初回は公開モデルを読み込みます。処理時間は端末や画像数により変わります。</p>
            {notice && <p className="status-notice" role="status">{notice}</p>}
            {error && <div className="error-notice" role="alert"><strong>処理を完了できませんでした</strong><p>{error}</p></div>}
          </section>
          <section className={`panel result-panel${result ? ' has-result' : ''}`} id="result" aria-labelledby="result-heading">
            <div className="section-title"><span className="step-number">03</span><div><h2 id="result-heading">{result ? 'モデルを書き出せます' : '結果を確認して、持ち出す'}</h2><p>{result ? '書き出したONNXを読み直し、学習後のモデルと出力が一致することを確認しました。' : '学習が完了すると、評価結果とダウンロードがここに表示されます。'}</p></div></div>
            {result ? <>
              <div className="verified-badge"><Icon kind="check" /><span>完全なONNXモデル · 独立した再読込検証 OK</span></div>
              <div className="result-metrics"><div className="accuracy-metric"><span>評価用画像の正解率</span><strong>{(result.summary.accuracy * 100).toFixed(1)}<small>%</small></strong><p>{result.summary.predictions.filter(item => item.actual === item.predicted).length} / {result.summary.testCount}枚が正解</p></div><dl><div><dt>学習 / 評価</dt><dd>{result.summary.trainingCount}枚 / {result.summary.testCount}枚</dd></div><div><dt>学習回数</dt><dd>{result.summary.epochs}エポック</dd></div><div><dt>処理時間</dt><dd>{formatDuration(result.summary.durationMs)}</dd></div></dl></div>
              {hasSyntheticImages && <p className="sample-result-note">図形サンプルでの結果です。自然画像のベンチマークや、実用時の精度を表す数値ではありません。</p>}
              <button type="button" className="button button-primary download-button" onClick={downloadModel}><Icon kind="download" />モデル一式をダウンロード<span aria-hidden="true">ZIP</span></button>
              <p className="download-note">完全なONNX、クラス名、入出力仕様、評価レポート、Python実行例、出典・ライセンスを含みます。</p>
              <details className="verification-details"><summary>画像ごとの結果と数値検証</summary><div className="prediction-table-wrap"><table><thead><tr><th>評価画像</th><th>正解</th><th>予測</th></tr></thead><tbody>{result.summary.predictions.map(prediction => <tr key={prediction.id}><td>{samples.find(sample => sample.id === prediction.id)?.name ?? prediction.id}</td><td>{trimmedLabels[prediction.actual]}</td><td><span className={prediction.actual === prediction.predicted ? 'prediction-correct' : 'prediction-incorrect'}>{trimmedLabels[prediction.predicted]}{prediction.actual === prediction.predicted ? ' ✓' : ' · 不一致'}</span></td></tr>)}</tbody></table></div><dl className="verification-values"><div><dt>元モデルとの最大絶対誤差</dt><dd>{result.summary.nativeMaxAbsError.toExponential(3)}</dd></div><div><dt>ONNX再読込時の最大絶対誤差</dt><dd>{result.summary.exportMaxAbsError.toExponential(3)}</dd></div><div><dt>分類部分の重みの変化（L2）</dt><dd>{result.summary.weightDeltaL2.toExponential(3)}</dd></div><div><dt>最初 / 最後の学習損失</dt><dd>{result.summary.losses.at(0)?.toFixed(5) ?? '—'} / {result.summary.losses.at(-1)?.toFixed(5) ?? '—'}</dd></div><div><dt>実行バックエンド</dt><dd>{result.summary.backend}</dd></div><div><dt>ONNXファイル</dt><dd>{(result.summary.modelBytes / 1024 / 1024).toFixed(2)} MiB</dd></div><div className="hash-value"><dt>モデルのSHA-256</dt><dd>{result.summary.modelSha256}</dd></div></dl><p>この画面の検証はブラウザ内の独立したONNX実行との比較です。Python環境での比較用データは評価レポートに含まれます。</p></details>
            </> : <div className="result-empty"><div className="result-icon"><Icon kind="download" /></div><h3>学習したモデルを、自分の環境へ。</h3><p>事前学習済みの特徴抽出部分と、学習後の分類部分をまとめたONNXを書き出します。</p><span>ONNX + ラベル + 実行例 + 検証レポート</span></div>}
          </section>
        </div>
      </main>
      <footer className="site-footer"><span>Open Vision Builder <span className="footer-separator">/</span> Classification preview <span className="footer-separator">/</span> <a href={`${import.meta.env.BASE_URL}runtime/THIRD_PARTY_LICENSES.txt`} target="_blank" rel="noopener">ライセンス表示</a></span><span>データはこのタブを閉じると失われます。{(buildInfo?.commit || buildInfo?.sha) && <code>commit {(buildInfo.commit || buildInfo.sha)!.slice(0, 8)}</code>}{buildInfo?.pr && <span className="build-pr">PR #{buildInfo.pr}</span>}</span></footer>
    </div>
  );
}
