import { describe, expect, it } from 'vitest';
import { preprocessRgba } from '../src/ml/preprocess';

describe('image preprocessing', () => {
  it('normalizes RGB into NCHW and repeats the border of a one-pixel image', () => {
    const t = preprocessRgba(new Uint8ClampedArray([255, 0, 0, 255]), 1, 1, 2);
    expect(t.length).toBe(12);
    expect(Array.from(t.slice(0, 4))).toEqual([2.248908281326294, 2.248908281326294, 2.248908281326294, 2.248908281326294]);
    expect(t[4]).toBeCloseTo(-2.0357142857, 5);
    expect(t[8]).toBeCloseTo(-1.8044444444, 5);
  });
  it('interpolates at half-pixel centers without shifting the left boundary', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]);
    const t = preprocessRgba(rgba, 2, 1, 4);
    expect(t[0]).toBeCloseTo(-2.11790393, 5);
    expect(t[1]).toBeCloseTo(-1.02620087, 5);
    expect(t[2]).toBeCloseTo(1.15720524, 5);
    expect(t[3]).toBeCloseTo(2.24890830, 5);
  });
  it('composites transparent pixels on white before interpolation', () => {
    const t = preprocessRgba(new Uint8ClampedArray([0, 0, 0, 0]), 1, 1, 1);
    expect(t[0]).toBeCloseTo(2.24890830, 5);
    expect(t[1]).toBeCloseTo(2.42857143, 5);
    expect(t[2]).toBeCloseTo(2.64, 5);
  });
  it('rejects invalid image dimensions and inconsistent byte lengths', () => {
    expect(() => preprocessRgba(new Uint8ClampedArray(4), 0, 1)).toThrow();
    expect(() => preprocessRgba(new Uint8ClampedArray(4), 2, 1)).toThrow();
  });
});
