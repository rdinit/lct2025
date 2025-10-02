# Build Stage
FROM golang:1.23-alpine AS builder


# Copy go.mod and go.sum first to leverage Docker cache
COPY go.mod .
COPY go.sum .

# Download dependencies
RUN go mod download

# Copy the rest of the application source code
COPY . .

# Build the Go application
# CGO_ENABLED=0 disables CGO, creating a statically linked binary
# -o /app/main specifies the output binary name and path
# -ldflags="-s -w" reduces the binary size by stripping debug info
RUN cd cmd/main && CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-s -w" -o /main .

# Runtime Stage
FROM alpine:3.19

# Install ca-certificates for HTTPS communication if needed
#RUN apk add --no-cache ca-certificates


# Copy the built binary from the builder stage
COPY --from=builder /main .

COPY  cmd/main .

# Expose the port your Go application listens on (e.g., 8080)
#EXPOSE 8082

# Command to run the application
CMD ["./main"]
