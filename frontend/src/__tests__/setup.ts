import "@testing-library/jest-dom";

// Minimal EventSource polyfill for jsdom — native EventSource is not
// available in jsdom. Components and hooks that use EventSource get this stub;
// tests that exercise stream behaviour mock openTelemetryStream directly.
class MockEventSource {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 2;

  readyState = MockEventSource.CONNECTING;
  onopen: ((e: Event) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;

  private _listeners = new Map<string, Array<(e: Event) => void>>();

  constructor(public url: string) {}

  addEventListener(type: string, listener: (e: Event) => void): void {
    const list = this._listeners.get(type) ?? [];
    list.push(listener);
    this._listeners.set(type, list);
  }

  removeEventListener(type: string, listener: (e: Event) => void): void {
    const list = this._listeners.get(type) ?? [];
    this._listeners.set(
      type,
      list.filter((l) => l !== listener),
    );
  }

  dispatchEvent(event: Event): boolean {
    const list = this._listeners.get(event.type) ?? [];
    list.forEach((l) => l(event));
    return true;
  }

  close(): void {
    this.readyState = MockEventSource.CLOSED;
  }
}

Object.defineProperty(globalThis, "EventSource", {
  value: MockEventSource,
  writable: true,
});
