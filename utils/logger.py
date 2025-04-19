 
import os
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

class Logger:
    """Logging utility that handles TensorBoard and CSV logging for experiments."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        flush_interval: int = 10
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of the experiment, defaults to timestamp if None
            use_tensorboard: Whether to use TensorBoard for logging
            use_csv: Whether to log metrics to CSV file
            flush_interval: How often to flush data to disk (in steps)
        """
        self.experiment_name = experiment_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        self.use_tensorboard = use_tensorboard and SummaryWriter is not None
        self.use_csv = use_csv
        self.flush_interval = flush_interval
        self.step_counter = 0
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if needed
        self.writer = None
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            
        # Initialize CSV logging if needed
        self.csv_file = None
        self.csv_writer = None
        self.headers = None
        if self.use_csv:
            self.csv_path = os.path.join(self.log_dir, "metrics.csv")
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
        self.logged_data = []
        self.start_time = time.time()
        
        print(f"Logger initialized. Logging to: {self.log_dir}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for the current step.
        
        Args:
            metrics: Dictionary of metrics to log (name: value)
            step: Current step, uses internal counter if None
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1
            
        # Add timestamp to metrics
        metrics["timestamp"] = time.time() - self.start_time
        metrics["step"] = step
            
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(name, value, step)
            
        # Store for CSV logging
        self.logged_data.append(metrics)
        
        # Set headers if not already set
        if self.use_csv and self.headers is None:
            self.headers = list(metrics.keys())
            self.csv_writer.writerow(self.headers)
            
        # Flush to CSV if interval reached
        if self.use_csv and len(self.logged_data) >= self.flush_interval:
            self._flush_to_csv()
            
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None) -> None:
        """Log histogram data to TensorBoard"""
        if step is None:
            step = self.step_counter
            
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def log_figure(self, name: str, figure, step: Optional[int] = None) -> None:
        """Log matplotlib figure to TensorBoard"""
        if step is None:
            step = self.step_counter
            
        if self.writer is not None:
            self.writer.add_figure(name, figure, step)
            
    def _flush_to_csv(self) -> None:
        """Flush logged data to CSV file"""
        if not self.use_csv or not self.logged_data:
            return
            
        for entry in self.logged_data:
            row = [entry.get(header, "") for header in self.headers]
            self.csv_writer.writerow(row)
            
        self.csv_file.flush()
        self.logged_data = []
    
    def close(self) -> None:
        """Close all loggers and flush remaining data"""
        if self.writer is not None:
            self.writer.close()
            
        if self.use_csv:
            self._flush_to_csv()
            if self.csv_file is not None:
                self.csv_file.close()
                
        print(f"Logger closed. Data saved to: {self.log_dir}")