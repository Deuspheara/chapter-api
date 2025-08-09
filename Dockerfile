# Secure multi-stage build for Rust with distroless runtime
# Stage 1: Build static binary using musl for maximum compatibility
FROM rust:1-bookworm AS builder

# (glibc build – no cross musl targets required)

# Set the working directory
WORKDIR /app

# Copy manifests and source from subfolder
COPY chapter-api/Cargo.toml Cargo.toml
COPY chapter-api/Cargo.lock Cargo.lock
COPY chapter-api/src ./src

# Build the application (glibc)
RUN cargo build --release

# Strip step omitted (musl-strip not available in all builder variants)

# Stage 2: Slim Debian runtime
FROM debian:bookworm-slim

# Install CA certificates for HTTPS (reqwest/rustls)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder stage
COPY --from=builder /app/target/release/chapter-api /app

# Optional working directory
WORKDIR /

# Create and switch to non-root user
RUN useradd -r -u 10001 appuser
USER 10001:10001

# Default environment
ENV RUST_BACKTRACE=1
ENV CHAPTERS_DIR=/tmp/chapters
ENV PORT=3000

# Expose the application port
EXPOSE 3000

# Run the application
CMD ["/app"]
