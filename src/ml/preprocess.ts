const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/** Full-image, half-pixel bilinear resize. Transparent pixels are composited on white. */
export function preprocessRgba(rgba: Uint8ClampedArray, width: number, height: number, size = 224): Float32Array {
  if (![width, height, size].every(n => Number.isInteger(n) && n > 0) || size > 2048 || rgba.length !== width * height * 4) {
    throw new Error('Invalid image dimensions or RGBA data');
  }
  const result = new Float32Array(3 * size * size);
  const pixel = (x: number, y: number, c: number) => {
    const i = (Math.max(0, Math.min(height - 1, y)) * width + Math.max(0, Math.min(width - 1, x))) * 4;
    const alpha = rgba[i + 3] / 255;
    return rgba[i + c] * alpha + 255 * (1 - alpha);
  };
  for (let y = 0; y < size; y++) {
    const sy = (y + 0.5) * height / size - 0.5;
    const y0 = Math.floor(sy), fy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = (x + 0.5) * width / size - 0.5;
      const x0 = Math.floor(sx), fx = sx - x0;
      for (let c = 0; c < 3; c++) {
        const top = pixel(x0, y0, c) * (1 - fx) + pixel(x0 + 1, y0, c) * fx;
        const bottom = pixel(x0, y0 + 1, c) * (1 - fx) + pixel(x0 + 1, y0 + 1, c) * fx;
        const value = top * (1 - fy) + bottom * fy;
        result[c * size * size + y * size + x] = (value / 255 - MEAN[c]) / STD[c];
      }
    }
  }
  return result;
}

export function referenceRgba(width = 40, height = 32): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    rgba[i] = (x * 7 + y * 3) % 256;
    rgba[i + 1] = (x * 2 + y * 11) % 256;
    rgba[i + 2] = (x * 13 + y * 5) % 256;
    rgba[i + 3] = 255;
  }
  return rgba;
}
