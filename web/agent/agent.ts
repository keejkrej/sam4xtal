import { defineAgent } from "eve";

/**
 * Stub eve agent for sam4xtal.
 * Intentionally does nothing useful yet — placeholder for future
 * crystal-annotation / dataset-curation agent workflows.
 *
 * Identity is derived from the package/app name by eve (do not set `name`).
 */
export default defineAgent({
  model: "openai/gpt-5.4-mini",
});
