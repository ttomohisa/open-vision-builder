// Self-contained so Playwright can install it before the production App loads.
export function installWorkerObserver() {
  const Original = window.Worker;
  window.Worker = class extends Original {
    constructor(...args) {
      super(...args);
      this.addEventListener('message', event => {
        const message = event.data;
        if (message.type === 'result') window.__ovbResult = { summary: message.summary, report: message.report };
        if (message.type === 'error') window.__ovbFailure = message.message;
        if (message.type === 'progress') (window.__ovbProgress ??= []).push({ stage: message.stage, fraction: message.fraction, loss: message.loss });
      });
    }

    postMessage(message, ...rest) {
      if (message.type !== 'run') return super.postMessage(message, ...rest);
      window.__ovbRequestCapture = Promise.all(message.samples.map(async sample => {
        const hash = await crypto.subtle.digest('SHA-256', await sample.file.arrayBuffer());
        return { id: sample.id, name: sample.name, imageId: sample.name.replace(/\.png$/, ''), label: sample.label,
          split: sample.split, fileSha256: Array.from(new Uint8Array(hash), x => x.toString(16).padStart(2, '0')).join('') };
      })).then(samples => ({ labels: message.labels, epochs: message.epochs, learningRate: message.learningRate, samples }));
      window.__ovbForwarding = window.__ovbRequestCapture.then(async request => {
        // The host validates fixed image bytes, order, splits and recipe before
        // the real Worker can perform any model inference or training.
        await window.__ovbValidateRequest(request);
        super.postMessage(message, ...rest);
      }).catch(error => {
        window.__ovbFailure = String(error);
        this.dispatchEvent(new MessageEvent('message', { data: { type: 'error', id: message.id, message: String(error) } }));
      });
    }
  };
}
