import { defineTool } from "eve/tools";
import { z } from "zod";

/** Placeholder tool so the agent directory is complete but inert. */
export default defineTool({
  description: "No-op stub tool. Returns a fixed acknowledgement.",
  inputSchema: z.object({
    note: z.string().optional(),
  }),
  async execute({ note }) {
    return {
      ok: true,
      message: "sam4xtal eve agent is a stub and does nothing yet.",
      note: note ?? null,
    };
  },
});
