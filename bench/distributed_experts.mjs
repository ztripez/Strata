#!/usr/bin/env node
// Repeatable single-primary qualification. Run a worker separately; this harness
// never changes weights, routing, cache policy, network configuration or services.
import { spawn, execFileSync } from 'node:child_process';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createServer, createConnection } from 'node:net';
import { createInterface } from 'node:readline';
import { resolve, join } from 'node:path';
import { once } from 'node:events';
import { finished } from 'node:stream/promises';
import { performance } from 'node:perf_hooks';

const opt = { cases: 'baseline,zero,remote,disconnect,timeout,unavailable', repeats: '3', tokens: '64', timeout: '5000',
  prompts: 'tools/qwen35/parity_prompts.ids', 'prompt-count': '3' };
for (let i = 2; i < process.argv.length; i += 2) {
  const key = process.argv[i].replace(/^--/, '');
  if (!process.argv[i].startsWith('--') || !process.argv[i + 1]) throw new Error('Expected --key value');
  opt[key] = process.argv[i + 1];
}
for (const key of ['engine', 'model', 'out', 'worker', 'placement']) if (!opt[key]) throw new Error(`--${key} is required`);
const modes = opt.cases.split(',');
for (const mode of modes) if (!['baseline','zero','remote','disconnect','timeout','unavailable'].includes(mode)) throw new Error(`Unknown case ${mode}`);
for (const key of ['repeats','tokens','timeout','prompt-count']) if (!/^\d+$/.test(opt[key]) || +opt[key] < 1) throw new Error(`Invalid --${key}`);
const dir = resolve(opt.out);
if (existsSync(dir)) throw new Error(`Refusing to overwrite evidence directory ${dir}`);
mkdirSync(dir, { recursive: true });
const prompts = readFileSync(opt.prompts, 'utf8').trim().split(/\r?\n/).slice(0, +opt['prompt-count']).map(line => {
  if (!/^\d+(,\d+)*$/.test(line)) throw new Error('Invalid prompt token IDs');
  return line.split(',').map(Number);
});
const zero = join(dir,'zero-placement.txt'); writeFileSync(zero, '# zero ownership\n');
const base = ['--model', resolve(opt.model), '--max-context', '2048', '--chunk', '256', '--mtp', '0', '--expert-slots', '12', '--serve'];
const env = { ...process.env, STRATA_Q8_PACKED: '0', STRATA_Q36_NO_MMQ: '1' };
const result = { revision: execFileSync('git',['rev-parse','HEAD'],{encoding:'utf8'}).trim(),
  source_diff: execFileSync('git',['diff'],{encoding:'utf8'}), started_at: new Date().toISOString(),
  engine: resolve(opt.engine), model: resolve(opt.model), worker: opt.worker,
  environment: { STRATA_Q8_PACKED:'0', STRATA_Q36_NO_MMQ:'1' }, base_args: base, prompts, cases: [] };
const hash = createHash('sha256'); for await (const chunk of createReadStream(opt.engine)) hash.update(chunk);
result.engine_sha256 = hash.digest('hex');
writeFileSync(join(dir,'placement.txt'),readFileSync(opt.placement));
const save = () => writeFileSync(join(dir,'results.json'),JSON.stringify(result,null,2)+'\n'); save();

// Frame-aware fault relay: cut after 30 complete worker responses, or withhold
// the first EXECUTE. Neither case modifies the real worker or its weights.
async function faultRelay(mode) {
  let held = false, responses = 0; const sockets = new Set();
  const [host, port] = opt.worker.split(':');
  const server = createServer(client => {
    const upstream = createConnection({host,port:+port});
    for (const s of [client,upstream]) { sockets.add(s); s.on('error', () => {}); s.on('close',()=>{client.destroy();upstream.destroy();}); }
    function route(from,to,isResult) {
      let pending=Buffer.alloc(0);
      from.on('data',chunk=>{
        pending=Buffer.concat([pending,chunk]);
        while(pending.length>=24) {
          const length=24+pending.readUInt32LE(16); if(length>64*1024*1024+24) throw new Error('Invalid relay frame');
          if(pending.length<length)return;
          const frame=pending.subarray(0,length);pending=pending.subarray(length);
          const type=frame.readUInt16LE(6);
          if(mode==='timeout' && !isResult && type===5) held=true;
          if(mode==='disconnect' && isResult && type===6 && ++responses>30) {client.destroy();upstream.destroy();return;}
          if(!held)to.write(frame);
        }
      });
    }
    route(client,upstream,false);route(upstream,client,true);
  });
  server.listen(0,'127.0.0.1');await once(server,'listening');
  return { endpoint:`127.0.0.1:${server.address().port}`, close:()=>{for(const s of sockets)s.destroy();server.close();},
    info:()=>({withheld_execute:held,worker_responses:responses}) };
}
function lineQueue(stream, log) {
  const queue=[];let waiter,ended=false;
  const lines=createInterface({input:stream});
  lines.on('line',line=>{log.write(line+'\n');if(waiter){const f=waiter;waiter=null;f(line);}else queue.push(line);});
  lines.on('close',()=>{ended=true;if(waiter){const f=waiter;waiter=null;f(null);}});
  return async timeout=>{
    if(queue.length)return queue.shift();if(ended)return null;
    return await new Promise((res,rej)=>{
      const timer=setTimeout(()=>{waiter=null;rej(new Error(`Engine output deadline ${timeout} ms`));},timeout);
      waiter=line=>{clearTimeout(timer);res(line);};
    });
  };
}
try {
  for(const mode of modes) {
    console.log(`Starting ${mode}`);
    const record={mode,requests:[],started_at:new Date().toISOString()};result.cases.push(record);save();
    let relay;
    if(mode==='disconnect'||mode==='timeout')relay=await faultRelay(mode);
    let endpoint=relay?.endpoint??opt.worker;
    if(mode==='unavailable') { const s=createServer();s.listen(0,'127.0.0.1');await once(s,'listening');endpoint=`127.0.0.1:${s.address().port}`;await new Promise(res=>s.close(res)); }
    const args=[...base];
    if(mode!=='baseline')args.push('--expert-worker',endpoint,'--expert-worker-placement',mode==='zero'?zero:resolve(opt.placement),'--expert-worker-timeout-ms',opt.timeout);
    record.args=args;
    const stderr=createWriteStream(join(dir,`${mode}.stderr`));const stdout=createWriteStream(join(dir,`${mode}.stdout`));
    const proc=spawn(opt.engine,args,{env,stdio:['pipe','pipe','pipe']});
    const done=once(proc,'close'); proc.stderr.pipe(stderr); const next=lineQueue(proc.stdout,stdout);
    const start=performance.now();
    try {
      let ready=false;
      for(;;) {const line=await next(600000);if(line===null)break;if(line.startsWith('READY ')){ready=true;break;}}
      record.startup_ms=performance.now()-start;
      if(mode==='unavailable') {
        if(ready)throw new Error('Unavailable worker unexpectedly became ready');
        const [code]=await done;record.exit_code=code;if(code===0)throw new Error('Unavailable worker silently succeeded');
      } else {
        if(!ready)throw new Error('Engine exited before READY');
        for(let rep=0;rep<+opt.repeats;rep++)for(let p=0;p<prompts.length;p++) {
          const r={prompt:p,repetition:rep,warm:rep>0,token_ids:[],token_times_ms:[]};record.requests.push(r);
          const t0=performance.now();proc.stdin.write(`GEN ${opt.tokens} temperature=0 seed=1 ${prompts[p].join(',')}\n`);
          for(;;) {
            const line=await next(60000);if(line===null)throw new Error('Engine exited during request');
            if(line.startsWith('T ')) {const id=Number(line.slice(2));if(!Number.isInteger(id))throw new Error('Invalid token');r.token_ids.push(id);r.token_times_ms.push(performance.now()-t0);}
            if(line.startsWith('ERR'))throw new Error(line);
            if(line.startsWith('DONE ')) {
              const f=line.split(' ');r.done=line;r.wall_ms=performance.now()-t0;
              r.produced=+f[1];r.prompt_tokens=+f[2];r.prompt_ms=+f[3];r.engine_decode_ms=+f[4];r.finish=f[5];
              r.reused_tokens=+f[8];r.fresh_tokens=+f[14];
              if(r.produced!==r.token_ids.length||r.token_ids.length===0)throw new Error('Incomplete token stream');
              r.ttft_ms=r.token_times_ms[0];
              r.decode_tok_s=(r.produced-1)*1000/(r.token_times_ms.at(-1)-r.ttft_ms);
              r.prefill_tok_s=r.fresh_tokens*1000/r.prompt_ms;
              console.log(`${mode} rep=${rep} prompt=${p} tokens=${r.produced} decode=${r.decode_tok_s.toFixed(2)} TTFT=${r.ttft_ms.toFixed(1)}ms`);
              save();break;
            }
          }
        }
        proc.stdin.end();const [code]=await done;record.exit_code=code;if(code!==0)throw new Error(`Engine exit ${code}`);
      }
      if(relay)record.fault=relay.info();
      await finished(stderr);
      const text=readFileSync(join(dir,`${mode}.stderr`),'utf8');
      record.worker_summaries=text.split('\n').filter(s=>s.startsWith('strata-network-worker:'));
      record.hybrid_summaries=text.split('\n').filter(s=>s.includes('hybrid experts:')||s.includes('network entries'));
      if(['disconnect','timeout'].includes(mode) && !/falling back to local CPU/.test(text)) throw new Error('Fault case did not report local recomputation');
      if(mode==='remote' && !/remote_entries=[1-9]/.test(text))throw new Error('No real remote work was acknowledged');
      if(mode==='zero' && /remote_entries=[1-9]/.test(text))throw new Error('Zero ownership executed remotely');
      record.status='completed';
    } catch(error) {record.status='failed';record.error=String(error);proc.kill('SIGKILL');await done;throw error;}
    finally {relay?.close();stdout.end();stderr.end();save();}
  }
  const baseline=result.cases.find(c=>c.mode==='baseline');
  if(baseline)for(const c of result.cases.filter(c=>c.mode!=='baseline'))for(let i=0;i<c.requests.length;i++) {
    const b=baseline.requests[i]?.token_ids,g=c.requests[i].token_ids;if(!b)continue;
    c.requests[i].token_mismatches=g.reduce((n,v,j)=>n+(v!==b[j]),0)+Math.max(0,b.length-g.length);
    c.requests[i].first_token_difference=g.findIndex((v,j)=>v!==b[j]);
  }
  const zeroCase=result.cases.find(c=>c.mode==='zero');
  if(zeroCase?.requests.some(r=>r.token_mismatches!==undefined && r.token_mismatches!==0)) throw new Error('Zero ownership changed generated tokens');
  result.finished_at=new Date().toISOString();save();
  const rows=['case,repeat,prompt,generated,prefill_tok_s,decode_tok_s,ttft_ms,token_mismatches'];
  for(const c of result.cases)for(const r of c.requests)rows.push([c.mode,r.repetition,r.prompt,r.produced,r.prefill_tok_s,r.decode_tok_s,r.ttft_ms,r.token_mismatches??0].join(','));
  writeFileSync(join(dir,'results.csv'),rows.join('\n')+'\n');
} catch(error) {console.error(error);save();process.exitCode=1;}
