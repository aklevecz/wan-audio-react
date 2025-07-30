#!/bin/bash

echo "🔄 Restarting start_server.py..."

# Kill the existing process
echo "📋 Killing existing start_server.py process..."
pkill -f "start_server.py"

# Wait a moment for graceful shutdown
sleep 3

# Check if process is still running
if pgrep -f "start_server.py" > /dev/null; then
    echo "⚠️  Process still running, force killing..."
    pkill -9 -f "start_server.py"
    sleep 2
fi

# Start the server again with nohup
echo "🚀 Starting start_server.py with nohup..."
nohup python start_server.py > nohup.out 2>&1 &

# Get the new PID
sleep 1
NEW_PID=$(pgrep -f "start_server.py")

if [ ! -z "$NEW_PID" ]; then
    echo "✅ Server restarted successfully!"
    echo "📋 New PID: $NEW_PID"
    echo "📄 Logs: tail -f nohup.out"
else
    echo "❌ Failed to start server"
    exit 1
fi