FROM node:22-bullseye-slim

RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app

COPY package.json pnpm-lock.yaml /app/

ENV PUPPETEER_SKIP_DOWNLOAD=true

RUN pnpm install --frozen-lockfile

COPY ./ /app/

EXPOSE 5173

CMD ["pnpm", "dev", "--host", "0.0.0.0"]
