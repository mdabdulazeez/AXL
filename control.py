"""
Control module for AXL BCI system.

This module provides classes and functions for controlling external devices
based on classified EEG signals, including serial communication and command mapping.
"""

import time
import serial
import threading
import logging
import json
import numpy as np
from queue import Queue
from .config import *

# Configure logging
logging.basicConfig(level=logging.INFO if DEBUG_MODE else logging.WARNING)
logger = logging.getLogger(__name__)

class DeviceController:
    """Base class for device controllers."""
    
    def __init__(self):
        """Initialize the device controller."""
        self.is_connected = False
        self.command_queue = Queue()
        self.thread = None
    
    def connect(self):
        """Connect to the device."""
        raise NotImplementedError("Subclasses must implement connect method")
    
    def disconnect(self):
        """Disconnect from the device."""
        raise NotImplementedError("Subclasses must implement disconnect method")
    
    def send_command(self, command):
        """
        Send a command to the device.
        
        Parameters:
        -----------
        command : dict
            Command dictionary with at least an 'action' key
        """
        self.command_queue.put(command)
    
    def start(self):
        """Start the control thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Control thread is already running")
            return
        
        self.thread = threading.Thread(target=self._control_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Control thread started")
    
    def stop(self):
        """Stop the control thread."""
        if self.thread is None or not self.thread.is_alive():
            logger.warning("Control thread is not running")
            return
        
        # Stop the control loop
        self.is_connected = False
        
        # Wait for the thread to terminate
        self.thread.join(timeout=1.0)
        logger.info("Control thread stopped")
    
    def _control_loop(self):
        """Control loop to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _control_loop method")

class SerialController(DeviceController):
    """Controller for devices connected via serial port."""
    
    def __init__(self, port=DEVICE_PORT, baud=DEVICE_BAUD, 
                 command_map=None, command_interval=COMMAND_INTERVAL):
        """
        Initialize the serial controller.
        
        Parameters:
        -----------
        port : str
            Serial port name
        baud : int
            Baud rate
        command_map : dict or None
            Mapping from command actions to serial commands.
            If None, a default mapping will be used.
        command_interval : float
            Minimum interval between commands in seconds
        """
        super().__init__()
        self.port = port
        self.baud = baud
        self.command_interval = command_interval
        self.serial = None
        
        # Default command mapping if none provided
        if command_map is None:
            self.command_map = {
                COMMANDS['NONE']: b'N',
                COMMANDS['FORWARD']: b'F',
                COMMANDS['BACKWARD']: b'B',
                COMMANDS['LEFT']: b'L',
                COMMANDS['RIGHT']: b'R',
                COMMANDS['STOP']: b'S'
            }
        else:
            self.command_map = command_map
        
        # Store last command and timestamp
        self.last_command = None
        self.last_command_time = 0
    
    def connect(self):
        """Connect to the device via serial port."""
        if self.is_connected:
            logger.warning(f"Already connected to {self.port}")
            return
        
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=1)
            self.is_connected = True
            logger.info(f"Connected to {self.port} at {self.baud} baud")
        except Exception as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            self.is_connected = False
    
    def disconnect(self):
        """Disconnect from the serial port."""
        if not self.is_connected:
            logger.warning("Not connected")
            return
        
        try:
            # Send stop command before disconnecting
            self._send_raw_command(self.command_map[COMMANDS['STOP']])
            
            # Close the serial port
            self.serial.close()
            self.is_connected = False
            logger.info(f"Disconnected from {self.port}")
        except Exception as e:
            logger.error(f"Error disconnecting from {self.port}: {e}")
    
    def _send_raw_command(self, raw_command):
        """
        Send a raw command to the device.
        
        Parameters:
        -----------
        raw_command : bytes
            Raw command to send
        """
        if not self.is_connected:
            logger.warning("Not connected, cannot send command")
            return
        
        try:
            self.serial.write(raw_command)
            self.serial.flush()
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            # Try to reconnect
            self.disconnect()
            self.connect()
    
    def _control_loop(self):
        """Control loop for processing commands."""
        # Connect to the device
        self.connect()
        
        while self.is_connected:
            try:
                # Get command from queue with timeout
                command = self.command_queue.get(timeout=0.1)
                
                # Check if it's time to send a new command
                current_time = time.time()
                if current_time - self.last_command_time >= self.command_interval:
                    # Get the action from the command
                    action = command.get('action', COMMANDS['NONE'])
                    
                    # Only send if the command is different or it's been a while
                    if action != self.last_command or current_time - self.last_command_time > 5 * self.command_interval:
                        # Map the action to a raw command
                        raw_command = self.command_map.get(action)
                        if raw_command is not None:
                            self._send_raw_command(raw_command)
                            self.last_command = action
                            self.last_command_time = current_time
                            logger.debug(f"Sent command: {action}")
                        else:
                            logger.warning(f"Unknown action: {action}")
            
            except Exception as e:
                if not isinstance(e, TimeoutError):
                    logger.error(f"Error in control loop: {e}")
        
        # Disconnect when the loop exits
        self.disconnect()

class SimulatedController(DeviceController):
    """Simulated controller for testing without real hardware."""
    
    def __init__(self, visualize=True, command_interval=COMMAND_INTERVAL):
        """
        Initialize the simulated controller.
        
        Parameters:
        -----------
        visualize : bool
            Whether to visualize the commands
        command_interval : float
            Minimum interval between commands in seconds
        """
        super().__init__()
        self.visualize = visualize
        self.command_interval = command_interval
        
        # Store current state
        self.position = np.array([0.0, 0.0])  # x, y position
        self.orientation = 0.0  # angle in radians (0 = right, pi/2 = up)
        self.speed = 0.0  # current speed
        
        # Store last command and timestamp
        self.last_command = None
        self.last_command_time = 0
        
        # Store the path history for visualization
        self.path_history = [self.position.copy()]
        
        # Visualization window
        self.fig = None
        self.ax = None
        if self.visualize:
            self._init_visualization()
    
    def _init_visualization(self):
        """Initialize the visualization window."""
        try:
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(-10, 10)
            self.ax.set_ylim(-10, 10)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title('Simulated Device Control')
            self.ax.set_xlabel('X position')
            self.ax.set_ylabel('Y position')
            plt.ion()  # Turn on interactive mode
            plt.show()
        except ImportError:
            logger.warning("Matplotlib not found, visualization disabled")
            self.visualize = False
    
    def connect(self):
        """Connect to the simulated device."""
        self.is_connected = True
        logger.info("Connected to simulated device")
    
    def disconnect(self):
        """Disconnect from the simulated device."""
        self.is_connected = False
        logger.info("Disconnected from simulated device")
        
        # Close the visualization window if it exists
        if self.visualize and self.fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            except Exception:
                pass
    
    def _update_state(self, action, dt=0.1):
        """
        Update the device state based on the action.
        
        Parameters:
        -----------
        action : int
            Command action
        dt : float
            Time step in seconds
        """
        # Define movement parameters
        max_speed = 1.0
        turn_rate = np.pi / 4  # 45 degrees per second
        acceleration = 0.5
        deceleration = 1.0
        
        # Update speed and orientation based on the action
        if action == COMMANDS['FORWARD']:
            # Accelerate forward
            self.speed = min(self.speed + acceleration * dt, max_speed)
        elif action == COMMANDS['BACKWARD']:
            # Accelerate backward
            self.speed = max(self.speed - acceleration * dt, -max_speed)
        elif action == COMMANDS['STOP']:
            # Decelerate to stop
            if self.speed > 0:
                self.speed = max(self.speed - deceleration * dt, 0)
            else:
                self.speed = min(self.speed + deceleration * dt, 0)
        elif action == COMMANDS['LEFT']:
            # Turn left
            self.orientation += turn_rate * dt
        elif action == COMMANDS['RIGHT']:
            # Turn right
            self.orientation -= turn_rate * dt
        
        # Update position based on speed and orientation
        dx = self.speed * np.cos(self.orientation) * dt
        dy = self.speed * np.sin(self.orientation) * dt
        self.position[0] += dx
        self.position[1] += dy
        
        # Store the position for visualization
        self.path_history.append(self.position.copy())
        
        # Keep only the last 1000 positions
        if len(self.path_history) > 1000:
            self.path_history = self.path_history[-1000:]
    
    def _update_visualization(self):
        """Update the visualization window."""
        if not self.visualize or self.fig is None:
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Clear the axes
            self.ax.clear()
            
            # Plot the path
            path = np.array(self.path_history)
            self.ax.plot(path[:, 0], path[:, 1], 'b-', alpha=0.5)
            
            # Plot the current position and orientation
            self.ax.plot(self.position[0], self.position[1], 'ro')
            
            # Draw an arrow to indicate orientation
            arrow_length = 0.5
            dx = arrow_length * np.cos(self.orientation)
            dy = arrow_length * np.sin(self.orientation)
            self.ax.arrow(self.position[0], self.position[1], dx, dy,
                         head_width=0.2, head_length=0.3, fc='r', ec='r')
            
            # Add a text label for the current state
            state_text = f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f})\n"
            state_text += f"Orientation: {np.degrees(self.orientation):.1f}°\n"
            state_text += f"Speed: {self.speed:.2f}\n"
            state_text += f"Command: {self.last_command}"
            self.ax.text(0.02, 0.98, state_text, transform=self.ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Set axis limits to keep the device in view with some margin
            margin = 2.0
            x_min = min(path[-50:, 0]) - margin if len(path) > 50 else -10
            x_max = max(path[-50:, 0]) + margin if len(path) > 50 else 10
            y_min = min(path[-50:, 1]) - margin if len(path) > 50 else -10
            y_max = max(path[-50:, 1]) + margin if len(path) > 50 else 10
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            
            # Grid and labels
            self.ax.grid(True)
            self.ax.set_title('Simulated Device Control')
            self.ax.set_xlabel('X position')
            self.ax.set_ylabel('Y position')
            self.ax.set_aspect('equal')
            
            # Update the plot
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            self.visualize = False
    
    def _control_loop(self):
        """Control loop for the simulated device."""
        # Connect to the simulated device
        self.connect()
        
        last_update_time = time.time()
        
        while self.is_connected:
            try:
                # Get command from queue with timeout
                try:
                    command = self.command_queue.get(timeout=0.01)
                    # Get the action from the command
                    action = command.get('action', COMMANDS['NONE'])
                    self.last_command = action
                except:
                    # No new command, use the last one
                    action = self.last_command
                
                # Update the simulation state
                current_time = time.time()
                dt = current_time - last_update_time
                self._update_state(action, dt)
                last_update_time = current_time
                
                # Update the visualization
                if self.visualize:
                    self._update_visualization()
                
                # Sleep to maintain a reasonable update rate
                time.sleep(0.05)
            
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
        
        # Disconnect when the loop exits
        self.disconnect()

def create_controller(controller_type='simulated', **kwargs):
    """
    Factory function to create an appropriate controller.
    
    Parameters:
    -----------
    controller_type : str
        Type of controller ('serial' or 'simulated')
    **kwargs : dict
        Additional parameters for the controller
        
    Returns:
    --------
    controller : DeviceController
        The created controller
    """
    if controller_type == 'serial':
        return SerialController(**kwargs)
    elif controller_type == 'simulated':
        return SimulatedController(**kwargs)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}") 