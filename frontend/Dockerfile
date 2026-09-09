# PaperPilot frontend.
#
# The build output is static files, so the runtime image is nginx with no
# Node in it at all — about 50 MB rather than the ~400 MB a node runtime
# would cost, and with a smaller attack surface for something that only
# ever serves assets.

# ---------------------------------------------------------------- builder
FROM node:22-alpine AS builder

WORKDIR /app

# Dependencies are copied and installed before the source, so an edit to a
# component does not invalidate the npm layer.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# `npm run build` runs `tsc -b` first, so a type error fails the image
# build rather than shipping. That is the point of doing it here.
RUN npm run build

# ---------------------------------------------------------------- runtime
FROM nginx:1.27-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
