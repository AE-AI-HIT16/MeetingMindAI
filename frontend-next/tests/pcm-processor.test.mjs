import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const processorSource = readFileSync(
  new URL("../public/pcm-processor.js", import.meta.url),
  "utf8",
);

function createProcessor(sampleRate) {
  let ProcessorClass;

  class FakeAudioWorkletProcessor {
    constructor() {
      this.emitted = [];
      this.port = {
        onmessage: null,
        postMessage: (buffer) => {
          this.emitted.push(new Int16Array(buffer).slice());
        },
      };
    }
  }

  vm.runInNewContext(processorSource, {
    ArrayBuffer,
    AudioWorkletProcessor: FakeAudioWorkletProcessor,
    Float32Array,
    Int16Array,
    Math,
    registerProcessor: (_name, processorClass) => {
      ProcessorClass = processorClass;
    },
  });

  const processor = new ProcessorClass();
  processor.port.onmessage({
    data: { command: "init", sampleRate },
  });
  return processor;
}

function processOneSecond(sampleRate) {
  const processor = createProcessor(sampleRate);
  const input = new Float32Array(sampleRate);
  for (let index = 0; index < input.length; index += 1) {
    input[index] = Math.sin(index / 100);
  }

  for (let offset = 0; offset < input.length; offset += 128) {
    processor.process([[input.subarray(offset, offset + 128)]]);
  }

  const emittedSamples = processor.emitted.reduce(
    (total, chunk) => total + chunk.length,
    0,
  );
  return emittedSamples + processor._accumulator.length;
}

function processValues(sampleRate, input) {
  const processor = createProcessor(sampleRate);
  processor._chunkSamples = 1;
  for (let offset = 0; offset < input.length; offset += 128) {
    processor.process([[input.subarray(offset, offset + 128)]]);
  }
  return processor.emitted.map((chunk) => chunk[0]);
}

test("preserves exactly one second when resampling 48 kHz to 16 kHz", () => {
  assert.equal(processOneSecond(48_000), 16_000);
});

test("preserves fractional phase when resampling 44.1 kHz", () => {
  assert.equal(processOneSecond(44_100), 16_000);
});

test("continues source position across 128-sample worklet blocks", () => {
  const input = new Float32Array(256);
  for (let index = 0; index < input.length; index += 1) {
    input[index] = index / 1000;
  }

  const output = processValues(48_000, input);

  assert.equal(output.length, 86);
  assert.equal(output[42], Math.trunc(input[126] * 0x7fff));
  assert.equal(output[43], Math.trunc(input[129] * 0x7fff));
});
