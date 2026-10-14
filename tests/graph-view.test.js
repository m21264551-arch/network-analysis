import test from 'node:test';
import assert from 'node:assert/strict';
import {alignPartitionColours} from '../graph-view.js';

test('display alignment maximizes shared membership without relabelling partitions', () => {
  // A greedy assignment would take 0 for group 8 and leave group 9 unmatched.
  const primary={a:0,b:0,c:0,d:1,e:1,f:0,g:0};
  const secondary={a:8,b:8,c:8,d:8,e:8,f:9,g:9};
  const before=structuredClone({primary,secondary});
  assert.deepEqual(alignPartitionColours(primary,secondary),{8:1,9:0});
  assert.deepEqual({primary,secondary},before);
});
test('different partition counts keep distinct deterministic display colours', () => {
  const primary={a:0,b:0,c:1,d:1}, secondary={a:4,b:5,c:6,d:6};
  const mapping=alignPartitionColours(primary,secondary);
  assert.equal(new Set(Object.values(mapping)).size,3);
  assert.equal(mapping[6],1);
  assert.deepEqual(alignPartitionColours(primary,secondary),mapping);
});
