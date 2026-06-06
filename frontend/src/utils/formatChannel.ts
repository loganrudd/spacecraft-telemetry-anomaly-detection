/**
 * Format a raw channel ID for display.
 *
 * "channel_12" → "Ch. 12"
 * "channel_3"  → "Ch. 3"
 * Anything else passes through unchanged so unexpected IDs are still readable.
 */
export function formatChannel(id: string): string {
  const m = id.match(/^channel_(\d+)$/);
  return m ? `Ch. ${m[1]}` : id;
}
