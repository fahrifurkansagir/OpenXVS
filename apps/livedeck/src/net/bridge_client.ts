/**
 * Optional local WebSocket avionics bridge client (`ws://127.0.0.1:8765`).
 *
 * Binary wire framing matches `tools/livedeck_bridge/bridge.py`:
 *   - 0x01 || GDL-90 framed bytes     -> UDP 127.0.0.1:4000
 *   - 0x02 || MAVLink v2 framed bytes -> UDP 127.0.0.1:14550
 *
 * Never throws if the local bridge is not running; surfaces explicit connection status cleanly.
 */

export type BridgeConnectionState = "disconnected" | "connecting" | "connected" | "error";

export interface BridgeTelemetry {
  readonly state: BridgeConnectionState;
  readonly url: string;
  readonly gdl90Sent: number;
  readonly mavlinkSent: number;
  readonly droppedOffline: number;
  readonly lastError: string | null;
}

export class LocalBridgeClient {
  private ws: WebSocket | null = null;
  private state: BridgeConnectionState = "disconnected";
  private url: string;
  private gdl90Sent = 0;
  private mavlinkSent = 0;
  private droppedOffline = 0;
  private lastError: string | null = null;
  private listeners = new Set<(tel: BridgeTelemetry) => void>();

  constructor(url = "ws://127.0.0.1:8765") {
    this.url = url;
  }

  subscribe(fn: (tel: BridgeTelemetry) => void): () => void {
    this.listeners.add(fn);
    fn(this.getTelemetry());
    return () => {
      this.listeners.delete(fn);
    };
  }

  private emit(): void {
    const snap = this.getTelemetry();
    for (const fn of this.listeners) {
      fn(snap);
    }
  }

  connect(customUrl?: string): void {
    if (customUrl) {
      this.url = customUrl;
    }
    this.disconnect();
    this.state = "connecting";
    this.lastError = null;
    this.emit();

    try {
      const sock = new WebSocket(this.url);
      sock.binaryType = "arraybuffer";
      this.ws = sock;

      sock.onopen = () => {
        if (this.ws !== sock) return;
        this.state = "connected";
        this.lastError = null;
        this.emit();
      };

      sock.onerror = () => {
        if (this.ws !== sock) return;
        this.state = "error";
        this.lastError = `Unable to reach local bridge at ${this.url}`;
        this.emit();
      };

      sock.onclose = () => {
        if (this.ws !== sock) return;
        this.ws = null;
        if (this.state !== "error") {
          this.state = "disconnected";
        }
        this.emit();
      };
    } catch (err) {
      this.ws = null;
      this.state = "error";
      this.lastError = err instanceof Error ? err.message : String(err);
      this.emit();
    }
  }

  disconnect(): void {
    if (this.ws) {
      const old = this.ws;
      this.ws = null;
      try {
        old.close(1000, "Client disconnect");
      } catch {
        // Ignore close errors on already-closed sockets
      }
    }
    this.state = "disconnected";
    this.emit();
  }

  sendGdl90(frame: Uint8Array): boolean {
    return this._sendPrefixed(0x01, frame, "gdl90");
  }

  sendMavlink(frame: Uint8Array): boolean {
    return this._sendPrefixed(0x02, frame, "mavlink");
  }

  private _sendPrefixed(prefix: number, payload: Uint8Array, kind: "gdl90" | "mavlink"): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.droppedOffline += 1;
      return false;
    }
    try {
      const out = new Uint8Array(1 + payload.byteLength);
      out[0] = prefix;
      out.set(payload, 1);
      this.ws.send(out.buffer);
      if (kind === "gdl90") {
        this.gdl90Sent += 1;
      } else {
        this.mavlinkSent += 1;
      }
      return true;
    } catch (err) {
      this.droppedOffline += 1;
      this.lastError = err instanceof Error ? err.message : String(err);
      return false;
    }
  }

  getTelemetry(): BridgeTelemetry {
    return {
      state: this.state,
      url: this.url,
      gdl90Sent: this.gdl90Sent,
      mavlinkSent: this.mavlinkSent,
      droppedOffline: this.droppedOffline,
      lastError: this.lastError,
    };
  }
}
