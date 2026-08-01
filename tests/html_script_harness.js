'use strict';

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');

function createClassList() {
  const values = new Set();
  return {
    add: (...names) => names.forEach(name => values.add(name)),
    remove: (...names) => names.forEach(name => values.delete(name)),
    contains: name => values.has(name),
    toggle: (name, force) => {
      const enabled = force === undefined ? !values.has(name) : Boolean(force);
      if (enabled) values.add(name);
      else values.delete(name);
      return enabled;
    },
    toString: () => [...values].join(' '),
  };
}

function createContext2d() {
  const noop = () => {};
  return new Proxy({
    canvas: null,
    measureText: text => ({ width: String(text).length * 8 }),
    createImageData: (width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
    getImageData: (_x, _y, width, height) => ({ width, height, data: new Uint8ClampedArray(width * height * 4) }),
  }, {
    get(target, property) {
      if (property in target) return target[property];
      return noop;
    },
    set(target, property, value) {
      target[property] = value;
      return true;
    },
  });
}

function createElement(tagName = 'div') {
  const listeners = new Map();
  const context2d = createContext2d();
  const element = {
    tagName: String(tagName).toUpperCase(),
    style: {},
    dataset: {},
    classList: createClassList(),
    children: [],
    files: [],
    value: '',
    checked: false,
    disabled: false,
    textContent: '',
    innerHTML: '',
    width: 1200,
    height: 1200,
    clientWidth: 1200,
    clientHeight: 1200,
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    dispatchEvent(event) {
      for (const handler of listeners.get(event.type) || []) handler.call(element, event);
    },
    appendChild(child) {
      element.children.push(child);
      return child;
    },
    removeChild(child) {
      element.children = element.children.filter(item => item !== child);
    },
    click() { element.clicked = true; },
    focus() {},
    getContext(type) {
      if (type !== '2d') throw new Error(`Unsupported context: ${type}`);
      context2d.canvas = element;
      return context2d;
    },
    getBoundingClientRect() {
      return { left: 0, top: 0, width: element.clientWidth, height: element.clientHeight };
    },
    toDataURL() { return 'data:image/png;base64,'; },
  };
  return element;
}

function extractInlineScript(html) {
  const matches = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/gi)];
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one inline script, found ${matches.length}`);
  }
  return matches[0][1];
}

function loadHtmlScript(htmlPath, overrides = {}) {
  const absolutePath = path.resolve(htmlPath);
  const elements = new Map();
  const downloads = [];
  const alerts = [];
  const confirms = [];
  const storage = new Map();
  const documentListeners = new Map();

  const document = {
    body: createElement('body'),
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, createElement(id === 'mainCanvas' || id === 'canvas' ? 'canvas' : 'div'));
      return elements.get(id);
    },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    createElement(tagName) {
      const element = createElement(tagName);
      if (String(tagName).toLowerCase() === 'a') {
        element.click = () => downloads.push({ href: element.href, download: element.download, blob: createdUrls.get(element.href) });
      }
      return element;
    },
    addEventListener(type, handler) {
      if (!documentListeners.has(type)) documentListeners.set(type, []);
      documentListeners.get(type).push(handler);
    },
  };

  const localStorage = {
    getItem: key => storage.has(key) ? storage.get(key) : null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
    clear: () => storage.clear(),
  };

  const createdUrls = new Map();
  let urlCounter = 0;
  class FakeFileReader {
    readAsText(file) {
      queueMicrotask(() => this.onload?.({ target: { result: file.textContent ?? '' } }));
    }
    readAsDataURL(file) {
      queueMicrotask(() => this.onload?.({ target: { result: file.dataUrl ?? 'data:image/png;base64,' } }));
    }
  }
  class FakeImage {
    constructor() {
      this.width = 0;
      this.height = 0;
      this.naturalWidth = 0;
      this.naturalHeight = 0;
    }
    set src(value) {
      this._src = value;
      queueMicrotask(() => this.onload?.());
    }
    get src() { return this._src; }
  }

  const sandbox = {
    console,
    document,
    navigator: { clipboard: { read: async () => [] } },
    localStorage,
    sessionStorage: localStorage,
    crypto: webcrypto,
    Blob,
    TextEncoder,
    TextDecoder,
    URL: Object.assign(URL, {
      createObjectURL(blob) {
        const url = `blob:test-${++urlCounter}`;
        createdUrls.set(url, blob);
        return url;
      },
      revokeObjectURL(url) { createdUrls.delete(url); },
    }),
    URLSearchParams,
    FileReader: FakeFileReader,
    Image: FakeImage,
    alert: message => alerts.push(String(message)),
    confirm: message => {
      confirms.push(String(message));
      return true;
    },
    setTimeout,
    clearTimeout,
    queueMicrotask,
    requestAnimationFrame: callback => callback(0),
    cancelAnimationFrame: () => {},
    location: { search: '' },
    ...overrides,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;

  const context = vm.createContext(sandbox);
  const html = fs.readFileSync(absolutePath, 'utf8');
  new vm.Script(extractInlineScript(html), { filename: absolutePath }).runInContext(context);

  return {
    context,
    document,
    elements,
    downloads,
    alerts,
    confirms,
    storage,
    createdUrls,
    evaluate(expression) {
      return vm.runInContext(expression, context);
    },
    snapshot(expression) {
      return JSON.parse(JSON.stringify(vm.runInContext(expression, context)));
    },
    initialize() {
      context.window.onload?.();
    },
  };
}

module.exports = { createElement, extractInlineScript, loadHtmlScript };
