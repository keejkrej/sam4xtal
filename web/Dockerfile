# Next.js UI. Build context must be the repo root (needs samples/).
FROM node:22-bookworm-slim AS deps
WORKDIR /app/web
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile --config.minimumReleaseAge=0

FROM node:22-bookworm-slim AS builder
WORKDIR /app/web
RUN corepack enable && corepack prepare pnpm@latest --activate
COPY --from=deps /app/web/node_modules ./node_modules
COPY web/ ./
# Samples are resolved as ../samples from the web cwd
COPY samples /app/samples
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-bookworm-slim AS runner
WORKDIR /app/web
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    INFERENCE_URL=http://sidecar:9001

RUN groupadd --system --gid 1001 nodejs \
    && useradd --system --uid 1001 --gid nodejs nextjs

COPY --from=builder /app/samples /app/samples
COPY --from=builder /app/web/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/web/.next/static ./.next/static

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
