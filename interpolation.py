import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
import json
import logging
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)

class SpatialInterpolator:
    """
    Enhanced spatial interpolation utilities for creating smooth contour surfaces.
    Includes GeoTIFF-like rasterization with smooth interpolation.
    """
    
    def __init__(self):
        self.methods = ['idw', 'linear', 'cubic', 'nearest', 'rbf', 'raster_idw']
        self._cache = {}
    
    def interpolate_grid(self, points, bounds, grid_size=50, method='idw', power=2, smooth=True):
        """
        Create interpolated grid from point data with enhanced algorithms.
        Now includes GeoTIFF-like rasterization approach.
        """
        try:
            # Validate input
            if not points or len(points) < 3:
                logger.warning("Insufficient points for interpolation")
                return None
            
            # Convert to numpy array for efficiency
            points_array = np.array(points, dtype=np.float64)
            
            # Extract coordinates and values
            lats = points_array[:, 0]
            lons = points_array[:, 1]
            values = points_array[:, 2]
            
            # Filter out invalid values more efficiently
            valid_mask = (values > 0) & np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
            
            if np.sum(valid_mask) < 3:
                logger.warning("Insufficient valid points after filtering")
                return None
            
            lats = lats[valid_mask]
            lons = lons[valid_mask]
            values = values[valid_mask]
            
            # Log statistics
            logger.info(f"Interpolating {len(values)} valid points using {method} method")
            logger.info(f"Value range: {values.min():.1f} - {values.max():.1f} ppb")
            
            # Adaptive grid sizing based on data density and bounds
            grid_size = self._calculate_optimal_grid_size(bounds, len(values), grid_size)
            
            # Create regular grid
            lat_range = np.linspace(bounds['min_lat'], bounds['max_lat'], grid_size)
            lon_range = np.linspace(bounds['min_lon'], bounds['max_lon'], grid_size)
            lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
            
            # Perform interpolation based on method
            if method == 'idw':
                interpolated_grid = self._idw_interpolation_optimized(
                    lats, lons, values, lat_grid, lon_grid, power
                )
            elif method == 'rbf':
                interpolated_grid = self._rbf_interpolation(
                    lats, lons, values, lat_grid, lon_grid
                )
            elif method == 'raster_idw':
                # GeoTIFF-like approach with smooth IDW interpolation
                interpolated_grid = self._raster_idw_interpolation(
                    lats, lons, values, lat_grid, lon_grid, power
                )
            else:
                # Use scipy griddata for other methods
                points_2d = np.column_stack([lons, lats])  # Note: lon, lat order for griddata
                grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
                
                interpolated_values = griddata(
                    points_2d, values, grid_points, 
                    method=method, 
                    fill_value=np.nan
                )
                interpolated_grid = interpolated_values.reshape(lat_grid.shape)
            
            # Handle edge cases and fill gaps
            interpolated_grid = self._fill_gaps_enhanced(interpolated_grid, values, method)
            
            # Apply smoothing if requested
            if smooth and method != 'nearest':
                sigma = self._get_optimal_smoothing_sigma(method, grid_size)
                interpolated_grid = gaussian_filter(interpolated_grid, sigma=sigma)
            
            # Calculate statistics
            valid_grid_values = interpolated_grid[~np.isnan(interpolated_grid)]
            
            if len(valid_grid_values) == 0:
                logger.warning("No valid interpolated values")
                return None
            
            # Create output data structure
            result = {
                'grid': interpolated_grid.tolist(),
                'lat_range': lat_range.tolist(),
                'lon_range': lon_range.tolist(),
                'bounds': bounds,
                'method': method,
                'original_points': len(points),
                'valid_points': len(lats),
                'grid_size': grid_size,
                'value_range': {
                    'min': float(np.nanmin(valid_grid_values)),
                    'max': float(np.nanmax(valid_grid_values)),
                    'mean': float(np.nanmean(valid_grid_values)),
                    'median': float(np.nanmedian(valid_grid_values)),
                    'std': float(np.nanstd(valid_grid_values))
                },
                'coverage': float(len(valid_grid_values) / (grid_size * grid_size)),
                'quality_metrics': self._calculate_quality_metrics(interpolated_grid, lats, lons, values)
            }
            
            logger.info(f"Interpolation complete. Coverage: {result['coverage']:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Interpolation error: {e}", exc_info=True)
            return None
    
    def _calculate_optimal_grid_size(self, bounds, num_points, requested_size):
        """Calculate optimal grid size based on data density and bounds."""
        # Calculate area
        lat_diff = bounds['max_lat'] - bounds['min_lat']
        lon_diff = bounds['max_lon'] - bounds['min_lon']
        area = lat_diff * lon_diff
        
        # Calculate point density (points per square degree)
        density = num_points / area if area > 0 else num_points
        
        # Adaptive sizing based on density
        if density > 500:  # High density
            optimal_size = min(100, max(60, requested_size))
        elif density > 100:  # Medium density
            optimal_size = min(80, max(40, requested_size))
        else:  # Low density
            optimal_size = min(60, max(30, requested_size))
        
        logger.info(f"Adaptive grid sizing: {num_points} points, density: {density:.1f} pts/deg², grid: {optimal_size}x{optimal_size}")
        return optimal_size
    
    def _get_optimal_smoothing_sigma(self, method, grid_size):
        """Get optimal smoothing parameter based on method and grid size."""
        base_sigma = 1.0
        
        if method == 'idw':
            return base_sigma * (grid_size / 50.0)
        elif method == 'raster_idw':
            return base_sigma * 1.5 * (grid_size / 50.0)  # More smoothing for raster-like
        elif method == 'linear':
            return base_sigma * 0.5
        else:
            return base_sigma
    
    def _raster_idw_interpolation(self, x_points, y_points, values, x_grid, y_grid, power=2):
        """
        GeoTIFF-like rasterization with smooth IDW interpolation.
        Similar to the approach in paste.txt but with continuous interpolation.
        """
        # Start with IDW interpolation
        interpolated_grid = self._idw_interpolation_optimized(
            x_points, y_points, values, x_grid, y_grid, power
        )
        
        # Apply additional processing for raster-like smoothness
        # Use distance-weighted local averaging for smoother transitions
        grid_shape = interpolated_grid.shape
        smoothed_grid = np.copy(interpolated_grid)
        
        # Apply local smoothing with distance weights
        kernel_size = max(3, min(7, grid_shape[0] // 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create distance-based kernel
        center = kernel_size // 2
        y_kernel, x_kernel = np.ogrid[-center:center+1, -center:center+1]
        kernel_distances = np.sqrt(x_kernel*x_kernel + y_kernel*y_kernel)
        kernel_weights = 1.0 / (1.0 + kernel_distances)
        kernel_weights /= np.sum(kernel_weights)
        
        # Apply convolution-like smoothing
        for i in range(center, grid_shape[0] - center):
            for j in range(center, grid_shape[1] - center):
                if not np.isnan(interpolated_grid[i, j]):
                    # Get local neighborhood
                    neighborhood = interpolated_grid[i-center:i+center+1, j-center:j+center+1]
                    valid_mask = ~np.isnan(neighborhood)
                    
                    if np.sum(valid_mask) > 0:
                        # Apply weighted average
                        valid_weights = kernel_weights[valid_mask]
                        valid_values = neighborhood[valid_mask]
                        if len(valid_values) > 0:
                            smoothed_grid[i, j] = np.average(valid_values, weights=valid_weights)
        
        return smoothed_grid
    
    def _idw_interpolation_optimized(self, x_points, y_points, values, x_grid, y_grid, power=2):
        """
        Enhanced IDW interpolation with adaptive distance weighting.
        """
        # Flatten grid for vectorized operations
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        # Stack points for distance calculation
        points = np.column_stack([x_points, y_points])
        grid_points = np.column_stack([x_flat, y_flat])
        
        # Calculate all distances at once using cdist
        distances = cdist(grid_points, points, metric='euclidean')
        
        # Handle exact matches (distance = 0)
        exact_matches = distances < 1e-10
        
        # Enhanced distance weighting with adaptive radius
        mean_distance = np.mean(distances[distances > 0])
        adaptive_factor = 1.0 + (mean_distance * 0.1)  # Adjust based on data scale
        
        # Calculate weights with adaptive power
        effective_power = power * adaptive_factor
        with np.errstate(divide='ignore'):
            weights = 1.0 / (distances ** effective_power)
            weights[exact_matches] = 0  # Will be handled separately
        
        # Apply distance-based cutoff to reduce noise from very distant points
        max_distance = np.percentile(distances[distances > 0], 90)
        distant_mask = distances > max_distance
        weights[distant_mask] *= 0.1  # Reduce influence of very distant points
        
        # Normalize weights
        weight_sums = np.sum(weights, axis=1)
        weight_sums[weight_sums == 0] = 1  # Avoid division by zero
        weights = weights / weight_sums[:, np.newaxis]
        
        # Perform interpolation
        interpolated_flat = np.sum(weights * values, axis=1)
        
        # Handle exact matches
        for i in range(len(interpolated_flat)):
            exact_idx = np.where(exact_matches[i])[0]
            if len(exact_idx) > 0:
                interpolated_flat[i] = values[exact_idx[0]]
        
        # Reshape back to grid
        return interpolated_flat.reshape(x_grid.shape)
    
    def _fill_gaps_enhanced(self, grid, original_values, method):
        """
        Enhanced gap filling with method-specific approaches.
        """
        if not np.any(np.isnan(grid)):
            return grid
        
        # Get indices of valid and invalid points
        valid_mask = ~np.isnan(grid)
        
        if not np.any(valid_mask):
            # If entire grid is NaN, fill with mean of original values
            return np.full_like(grid, np.mean(original_values))
        
        # Method-specific gap filling
        if method in ['idw', 'raster_idw']:
            # Use distance-weighted interpolation for gaps
            return self._fill_gaps_idw(grid, valid_mask)
        else:
            # Use nearest neighbor for other methods
            return self._fill_gaps_nearest(grid, valid_mask)
    
    def _fill_gaps_idw(self, grid, valid_mask):
        """Fill gaps using IDW from nearby valid points."""
        filled_grid = np.copy(grid)
        
        # Get coordinates of valid and invalid points
        y_indices, x_indices = np.indices(grid.shape)
        y_valid = y_indices[valid_mask]
        x_valid = x_indices[valid_mask]
        values_valid = grid[valid_mask]
        
        y_invalid = y_indices[~valid_mask]
        x_invalid = x_indices[~valid_mask]
        
        if len(y_invalid) > 0 and len(y_valid) > 0:
            # Calculate distances from invalid to valid points
            invalid_points = np.column_stack([y_invalid, x_invalid])
            valid_points = np.column_stack([y_valid, x_valid])
            
            distances = cdist(invalid_points, valid_points, metric='euclidean')
            
            # Use only nearby points for efficiency
            max_neighbors = min(10, len(y_valid))
            
            for i, (y_inv, x_inv) in enumerate(zip(y_invalid, x_invalid)):
                # Get nearest neighbors
                neighbor_distances = distances[i]
                nearest_indices = np.argsort(neighbor_distances)[:max_neighbors]
                
                # Calculate IDW weights
                nearest_distances = neighbor_distances[nearest_indices]
                nearest_distances[nearest_distances == 0] = 1e-10  # Avoid division by zero
                
                weights = 1.0 / (nearest_distances ** 2)
                weights /= np.sum(weights)
                
                # Interpolate value
                nearest_values = values_valid[nearest_indices]
                filled_grid[y_inv, x_inv] = np.sum(weights * nearest_values)
        
        return filled_grid
    
    def _fill_gaps_nearest(self, grid, valid_mask):
        """Fill gaps using nearest neighbor interpolation."""
        from scipy.interpolate import NearestNDInterpolator
        
        # Get coordinates of valid points
        y_valid, x_valid = np.where(valid_mask)
        values_valid = grid[valid_mask]
        
        # Create nearest neighbor interpolator
        nn_interp = NearestNDInterpolator(
            list(zip(y_valid, x_valid)), 
            values_valid
        )
        
        # Get coordinates of invalid points
        y_invalid, x_invalid = np.where(~valid_mask)
        
        # Fill invalid points
        if len(y_invalid) > 0:
            filled_values = nn_interp(y_invalid, x_invalid)
            grid[~valid_mask] = filled_values
        
        return grid
    
    def _calculate_quality_metrics(self, grid, x_points, y_points, values):
        """Calculate quality metrics for the interpolation."""
        try:
            # Calculate smoothness (lower variance in gradients = smoother)
            gy, gx = np.gradient(grid)
            gradient_variance = np.nanvar(np.sqrt(gx**2 + gy**2))
            
            # Calculate coverage (percentage of non-NaN values)
            total_cells = grid.shape[0] * grid.shape[1]
            valid_cells = np.sum(~np.isnan(grid))
            coverage = valid_cells / total_cells
            
            # Calculate interpolation accuracy (if we have enough points)
            accuracy_score = None
            if len(values) > 10:
                # Use subset of points for cross-validation
                sample_size = min(len(values) // 3, 20)
                sample_indices = np.random.choice(len(values), sample_size, replace=False)
                
                # For simplicity, use mean absolute error as accuracy metric
                errors = []
                for idx in sample_indices:
                    # Find nearest grid point
                    lat_diff = np.abs(np.linspace(0, 1, grid.shape[0]) - y_points[idx])
                    lon_diff = np.abs(np.linspace(0, 1, grid.shape[1]) - x_points[idx])
                    lat_idx = np.argmin(lat_diff)
                    lon_idx = np.argmin(lon_diff)
                    
                    if not np.isnan(grid[lat_idx, lon_idx]):
                        error = abs(grid[lat_idx, lon_idx] - values[idx])
                        errors.append(error)
                
                if errors:
                    accuracy_score = np.mean(errors)
            
            return {
                'smoothness': float(1.0 / (1.0 + gradient_variance)) if gradient_variance > 0 else 1.0,
                'coverage': float(coverage),
                'accuracy': float(accuracy_score) if accuracy_score is not None else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quality metrics: {e}")
            return {'smoothness': None, 'coverage': None, 'accuracy': None}