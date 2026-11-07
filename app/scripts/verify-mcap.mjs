// Validate a LiveDirt MCAP: indexed read, schemas/channels, message counts, and decode one message per topic.
import fs from 'node:fs';
import { McapIndexedReader } from '@mcap/core';
const path = process.argv[2];
const buf = fs.readFileSync(path);
const readable = { size: async () => BigInt(buf.byteLength), read: async (offset, size) => new Uint8Array(buf.buffer, buf.byteOffset + Number(offset), Number(size)) };
const reader = await McapIndexedReader.Initialize({ readable });
const stats = reader.statistics;
console.log('messages', stats?.messageCount, 'channels', reader.channelsById.size, 'schemas', reader.schemasById.size, 'duration s', Number((stats.messageEndTime - stats.messageStartTime)) / 1e9);
const counts = {}; const first = {};
for await (const m of reader.readMessages()) {
  const ch = reader.channelsById.get(m.channelId); const sc = reader.schemasById.get(ch.schemaId);
  counts[ch.topic] = (counts[ch.topic] ?? 0) + 1;
  if (!first[ch.topic]) { const j = JSON.parse(new TextDecoder().decode(m.data)); first[ch.topic] = { schema: sc.name, keys: Object.keys(j).join(','), ...(j.point_stride ? { points: Buffer.from(j.data, 'base64').length / j.point_stride } : {}), ...(j.format ? { jpegBytes: Buffer.from(j.data, 'base64').length, jpegMagic: Buffer.from(j.data, 'base64').subarray(0, 3).toString('hex') } : {}), ...(j.column_count ? { grid: `${j.column_count}x${Buffer.from(j.data, 'base64').length / j.row_stride}` } : {}) }; }
}
console.log(JSON.stringify({ counts, first }, null, 1));
