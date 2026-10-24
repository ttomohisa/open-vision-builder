export interface DecodedImage { rgba: Uint8ClampedArray; width: number; height: number }

const MAX_PIXELS = 16_000_000;
const MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const PACKED_RGB = new Set(['RGBA', 'RGBX', 'BGRA', 'BGRX']);
const OPAQUE_YUV = new Set(['I420', 'I422', 'I444', 'NV12', 'I420P10', 'I422P10', 'I444P10', 'I420P12', 'I422P12', 'I444P12']);
// These orientation properties are supported by the qualified Chromium version,
// but are not yet declared by the pinned TypeScript DOM library.
type OrientedFrame = VideoFrame & { readonly rotation: number; readonly flip: boolean };

function checkDimensions(width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('画像のサイズを読み取れませんでした。');
  if (width * height > MAX_PIXELS) throw new Error('画像は1600万画素以下にしてください。');
}

/** Read only the container's size fields, before allocating decoded pixels. */
function checkEncodedDimensions(bytes: Uint8Array, type: string): void {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset: number, text: string) => [...text].every((letter, i) => bytes[offset + i] === letter.charCodeAt(0));
  if (type === 'image/png' && bytes.length >= 24 && bytes[0] === 137 && tag(1, 'PNG\r\n\x1a\n') && tag(12, 'IHDR')) {
    checkDimensions(view.getUint32(16), view.getUint32(20));
    return;
  }
  if (type === 'image/jpeg' && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    while (offset < bytes.length) {
      if (bytes[offset++] !== 0xff) break;
      while (bytes[offset] === 0xff) offset++;
      const marker = bytes[offset++];
      if (marker === 0xd9 || marker === 0xda) break;
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
      if (offset + 2 > bytes.length) break;
      const length = view.getUint16(offset);
      if (length < 2 || offset + length > bytes.length) break;
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker) && length >= 8) {
        checkDimensions(view.getUint16(offset + 5), view.getUint16(offset + 3));
        return;
      }
      offset += length;
    }
  }
  if (type === 'image/webp' && bytes.length >= 20 && tag(0, 'RIFF') && tag(8, 'WEBP')) {
    for (let offset = 12; offset + 8 <= bytes.length;) {
      const length = view.getUint32(offset + 4, true);
      const start = offset + 8;
      if (start + length > bytes.length) break;
      if (tag(offset, 'VP8X') && length >= 10) {
        const uint24 = (i: number) => bytes[i] + bytes[i + 1] * 256 + bytes[i + 2] * 65536;
        checkDimensions(uint24(start + 4) + 1, uint24(start + 7) + 1);
        return;
      }
      if (tag(offset, 'VP8L') && length >= 5 && bytes[start] === 0x2f) {
        const bits = view.getUint32(start + 1, true);
        checkDimensions((bits & 0x3fff) + 1, ((bits >>> 14) & 0x3fff) + 1);
        return;
      }
      if (tag(offset, 'VP8 ') && length >= 10 && bytes[start + 3] === 0x9d && bytes[start + 4] === 0x01 && bytes[start + 5] === 0x2a) {
        checkDimensions(view.getUint16(start + 6, true) & 0x3fff, view.getUint16(start + 8, true) & 0x3fff);
        return;
      }
      offset = start + length + (length & 1);
    }
  }
  throw new Error('画像のヘッダーを読み取れませんでした。PNG・JPEG・WebPのファイルを選んでください。');
}

async function decodeOpaqueBitmap(file: Blob): Promise<DecodedImage> {
  // Chromium's ImageDecoder YUV copy uses different chroma upsampling from
  // Pillow. Its bitmap decoder agrees with our JPEG and opaque WebP fixtures.
  // Use this only for JPEG or a frame proven opaque by its native pixel format;
  // opaque pixels cannot lose RGB to a Canvas premultiplication round trip.
  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: 'none', premultiplyAlpha: 'none', imageOrientation: 'from-image',
  });
  let canvas: OffscreenCanvas | undefined;
  try {
    checkDimensions(bitmap.width, bitmap.height);
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('このブラウザでは画像処理を開始できません。');
    context.drawImage(bitmap, 0, 0);
    return { rgba: context.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
    if (canvas) canvas.width = canvas.height = 1;
  }
}

/** Decode unpremultiplied bytes, ignore ICC conversion, and apply EXIF orientation. */
export async function decodeImage(file: Blob): Promise<DecodedImage> {
  if (!MIME_TYPES.has(file.type)) throw new Error('PNG・JPEG・WebPの画像を選んでください。');
  if (file.size > 20 * 1024 * 1024) throw new Error('画像は1枚20MiB以下にしてください。');
  const data = await file.arrayBuffer();
  checkEncodedDimensions(new Uint8Array(data), file.type);
  if (file.type === 'image/jpeg') return decodeOpaqueBitmap(file);
  if (typeof ImageDecoder === 'undefined' || !await ImageDecoder.isTypeSupported(file.type)) {
    throw new Error('画像を正確に読み込むには、ImageDecoderに対応したChromeまたはEdgeとHTTPSが必要です。');
  }
  const decoder = new ImageDecoder({ data, type: file.type, colorSpaceConversion: 'none', preferAnimation: false });
  let frame: OrientedFrame | undefined;
  try {
    frame = (await decoder.decode({ frameIndex: 0, completeFramesOnly: true })).image as OrientedFrame;
    checkDimensions(frame.codedWidth, frame.codedHeight);
    const rect = frame.visibleRect;
    if (!rect) throw new Error('画像の表示領域を読み取れませんでした。');
    checkDimensions(rect.width, rect.height);
    if (![0, 90, 180, 270].includes(frame.rotation) || typeof frame.flip !== 'boolean') {
      throw new Error('このブラウザでは画像の回転・反転を正確に読み取れません。最新のChromeまたはEdgeを使ってください。');
    }
    const native = frame.format !== null && PACKED_RGB.has(frame.format);
    if (!native) {
      if (frame.format === null || !OPAQUE_YUV.has(frame.format)) {
        throw new Error(`画像の画素形式 ${frame.format ?? 'unknown'} は精度を保って読み込めません。8bit PNGまたはJPEGに変換してください。`);
      }
      return await decodeOpaqueBitmap(file);
    }
    // Supplying format, even the same format, makes Chromium convert through a
    // premultiplied surface. Copy native packed RGB bytes without that option.
    const bytes = new Uint8Array(frame.allocationSize());
    const layout = await frame.copyTo(bytes);
    const plane = layout[0];
    if (layout.length !== 1 || !plane || plane.stride < rect.width * 4 ||
        plane.offset + (rect.height - 1) * plane.stride + rect.width * 4 > bytes.length) {
      throw new Error('画像の画素配置を読み取れませんでした。');
    }
    const format = frame.format!;
    const blueFirst = format === 'BGRA' || format === 'BGRX';
    const opaque = format === 'RGBX' || format === 'BGRX';
    const rotated = frame.rotation === 90 || frame.rotation === 270;
    const width = rotated ? rect.height : rect.width;
    const height = rotated ? rect.width : rect.height;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < rect.height; y++) for (let x = 0; x < rect.width; x++) {
      let dx = x, dy = y;
      if (frame.rotation === 90) { dx = rect.height - 1 - y; dy = x; }
      else if (frame.rotation === 180) { dx = rect.width - 1 - x; dy = rect.height - 1 - y; }
      else if (frame.rotation === 270) { dx = y; dy = rect.width - 1 - x; }
      if (frame.flip) dx = width - 1 - dx;
      const source = plane.offset + y * plane.stride + x * 4;
      const target = (dy * width + dx) * 4;
      rgba[target] = bytes[source + (blueFirst ? 2 : 0)];
      rgba[target + 1] = bytes[source + 1];
      rgba[target + 2] = bytes[source + (blueFirst ? 0 : 2)];
      rgba[target + 3] = opaque ? 255 : bytes[source + 3];
    }
    return { rgba, width, height };
  } finally {
    frame?.close();
    decoder.close();
  }
}
