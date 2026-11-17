/**
 * slmbench extension for Pi (loaded with `pi -e` by `slmbench browse`): Pi's interactive interface stays on screen so the
 * agent can be watched while it browses pi.dev, and this extension reports what it does to slmbench, as `pi --mode json`
 * would.
 *
 * - Appends one JSON line per event to $SLMBENCH_PI_EVENTS: every message (the system one lists the tools Pi declares),
 *   every tool call and its result, and when the agent settles. slmbench scores the episode from these events and the page.
 * - Shows what is running (the site, or the task) in Pi's footer.
 * - Scored tasks ($SLMBENCH_PI_EXIT_ON_SETTLE=1): when the agent settles (it answered without a tool call), waits
 *   $SLMBENCH_PI_LINGER_MS so the answer can be read, then closes Pi. In the demo Pi stays open for the next message.
 */
import { appendFileSync } from "node:fs";

export default function (pi: any) {
  const path = process.env.SLMBENCH_PI_EVENTS;
  const linger = Number(process.env.SLMBENCH_PI_LINGER_MS || "1500");
  const label = process.env.SLMBENCH_PI_LABEL || "";
  const exitOnSettle = process.env.SLMBENCH_PI_EXIT_ON_SETTLE === "1";
  const emit = (event: object) => {
    if (!path) return;
    try {
      appendFileSync(path, JSON.stringify(event) + "\n");
    } catch {
      // the episode is scored from the page even if an event is lost
    }
  };

  pi.on("session_start", async (_event: any, ctx: any) => {
    if (label && ctx.hasUI) ctx.ui.setStatus("slmbench", label);
  });
  pi.on("message_start", async (event: any) => {
    emit({ type: "message_start", message: { role: event.message?.role } });
  });
  pi.on("message_end", async (event: any) => {
    emit({ type: "message_end", message: event.message });
  });
  pi.on("tool_execution_start", async (event: any) => {
    emit({ type: "tool_execution_start", toolName: event.toolName, args: event.args });
  });
  pi.on("tool_execution_end", async (event: any) => {
    emit({ type: "tool_execution_end", toolName: event.toolName, isError: event.isError, result: event.result });
  });
  pi.on("agent_settled", async (event: any, ctx: any) => {
    emit({ type: "agent_settled", aborted: !!event.aborted });
    if (exitOnSettle) setTimeout(() => ctx.shutdown(), linger);
  });
}
