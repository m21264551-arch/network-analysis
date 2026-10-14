// Match display colours, never the stored community labels or evaluation metrics.
export function alignPartitionColours(primary, secondary) {
  const left=[...new Set(Object.values(primary))].sort((a,b)=>a-b);
  const right=[...new Set(Object.values(secondary))].sort((a,b)=>a-b);
  const available=[...new Set([...left,...right])];
  const weights=right.map(group=>available.map(colour=>Object.keys(secondary).filter(id=>secondary[id]===group && primary[id]===colour).length));
  // These curated experiments have small partitions. Dynamic programming avoids
  // a greedy match that would give two communities the same display colour.
  const memo=new Map();
  function match(index,used) {
    if(index===right.length) return {score:0,choices:[]};
    const key=`${index}:${used}`; if(memo.has(key)) return memo.get(key);
    let best=null;
    available.forEach((colour,j)=>{
      if(used & (1<<j)) return;
      const tail=match(index+1,used | (1<<j));
      const candidate={score:weights[index][j]+tail.score,choices:[colour,...tail.choices]};
      if(!best || candidate.score>best.score) best=candidate;
    });
    memo.set(key,best);return best;
  }
  const result=match(0,0);
  return Object.fromEntries(right.map((group,i)=>[group,result.choices[i]]));
}
