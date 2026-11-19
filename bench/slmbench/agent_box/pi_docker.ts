/**
 * slmbench extension for Pi (loaded with `pi -e` by the agent mode): Pi's tools act on the SWE-bench task container, not on
 * the machine Pi runs on. read, write, edit and bash keep Pi's own definitions (prompts, schemas, output limits) and only
 * their operations go through `docker exec -i $SLMBENCH_CONTAINER` (the container has no network). Pi runs with its working
 * directory set to /testbed, the repository's path inside the container, so paths mean the same on both sides.
 *
 * Based on Pi's own example of remote tools (examples/extensions/ssh.ts).
 */
import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  type BashOperations,
  createBashTool,
  createEditTool,
  createReadTool,
  createWriteTool,
  type EditOperations,
  type ReadOperations,
  type WriteOperations,
} from "@earendil-works/pi-coding-agent";

const container = process.env.SLMBENCH_CONTAINER || "";
const q = (s: string) => "'" + s.replace(/'/g, "'\\''") + "'";

function run(command: string, input?: Buffer | string): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const child = spawn("docker", ["exec", "-i", container, "bash", "-c", command], { stdio: ["pipe", "pipe", "pipe"] });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    child.stdout.on("data", (d) => out.push(d));
    child.stderr.on("data", (d) => err.push(d));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) reject(new Error(Buffer.concat(err).toString().trim() || `exit code ${code}`));
      else resolve(Buffer.concat(out));
    });
    child.stdin.end(input ?? "");
  });
}

const readOps: ReadOperations = {
  readFile: (p) => run(`cat -- ${q(p)}`),
  access: (p) => run(`test -r ${q(p)}`).then(() => {}),
  detectImageMimeType: async () => null,
};

const writeOps: WriteOperations = {
  writeFile: (p, content) => run(`cat > ${q(p)}`, content).then(() => {}),
  mkdir: (dir) => run(`mkdir -p -- ${q(dir)}`).then(() => {}),
};

const editOps: EditOperations = {
  readFile: readOps.readFile,
  access: (p) => run(`test -r ${q(p)} && test -w ${q(p)}`).then(() => {}),
  writeFile: writeOps.writeFile,
};

const bashOps: BashOperations = {
  exec: (command, cwd, { onData, signal, timeout }) =>
    new Promise((resolve, reject) => {
      const child = spawn("docker", ["exec", "-w", cwd, container, "bash", "-c", command], { stdio: ["ignore", "pipe", "pipe"] });
      let timedOut = false;
      const timer = timeout ? setTimeout(() => { timedOut = true; child.kill(); }, timeout * 1000) : undefined;
      child.stdout.on("data", onData);
      child.stderr.on("data", onData);
      child.on("error", (e) => { if (timer) clearTimeout(timer); reject(e); });
      const onAbort = () => child.kill();
      signal?.addEventListener("abort", onAbort, { once: true });
      child.on("close", (code) => {
        if (timer) clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
        if (signal?.aborted) reject(new Error("aborted"));
        else if (timedOut) reject(new Error(`timeout:${timeout}`));
        else resolve({ exitCode: code });
      });
    }),
};

export default function (pi: ExtensionAPI) {
  if (!container) throw new Error("SLMBENCH_CONTAINER is not set");
  const cwd = process.cwd();
  pi.registerTool(createReadTool(cwd, { operations: readOps }));
  pi.registerTool(createWriteTool(cwd, { operations: writeOps }));
  pi.registerTool(createEditTool(cwd, { operations: editOps }));
  pi.registerTool(createBashTool(cwd, { operations: bashOps }));
  pi.on("user_bash", () => ({ operations: bashOps }));
}
