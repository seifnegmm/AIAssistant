# ---- Base ----
FROM node:20-slim AS base

WORKDIR /app

COPY package*.json ./
RUN npm install

# ---- Development ----
FROM base AS development

COPY . .

# Create .next dir and set permissions before switching user
RUN mkdir -p /app/.next && \
    groupadd -r appuser && useradd -r -g appuser -d /app appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

CMD ["npm", "run", "dev"]

# ---- Production ----
FROM base AS production

COPY . .
RUN npm run build

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 3000

CMD ["npm", "start"]
