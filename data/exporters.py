"""
Data Exporters

Export analysis results to Excel, CSV, and other formats.
Supports multiple sheets, formatting, and organized output.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings


class ExcelExporter:
    """
    Export analysis results to Excel with multiple sheets.

    Typical usage:
    - Sheet 1: Raw features (all extracted parameters per recording/block)
    - Sheet 2: Aggregated features per recording
    - Sheet 3: Statistical test results
    - Sheet 4: Model performance metrics

    Attributes:
        filepath (str): Output Excel file path
        sheets (Dict[str, pd.DataFrame]): Dictionary of sheet_name -> DataFrame
    """

    def __init__(self, filepath: str):
        """
        Initialize Excel exporter.

        Args:
            filepath: Path for output Excel file
        """
        self.filepath = Path(filepath)
        self.sheets: Dict[str, Dict[str, Any]] = {}

    def add_sheet(
        self,
        sheet_name: str,
        data: pd.DataFrame,
        index: bool = True
    ) -> None:
        """
        Add a sheet to the Excel file.

        Args:
            sheet_name: Name of the sheet
            data: DataFrame to export
            index: Whether to include DataFrame index
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be a pandas DataFrame, got {type(data)}")

        # Truncate sheet name if too long (Excel limit is 31 characters)
        if len(sheet_name) > 31:
            original_name = sheet_name
            sheet_name = sheet_name[:31]
            warnings.warn(
                f"Sheet name '{original_name}' truncated to '{sheet_name}' "
                f"(Excel limit: 31 characters)"
            )

        self.sheets[sheet_name] = {
            'data': data,
            'index': index
        }

    def add_features_sheet(
        self,
        features_df: pd.DataFrame,
        sheet_name: str = 'Features'
    ) -> None:
        """
        Add features sheet with formatting.

        Args:
            features_df: DataFrame with extracted features
            sheet_name: Sheet name
        """
        self.add_sheet(sheet_name, features_df, index=True)

    def add_statistical_results_sheet(
        self,
        stats_df: pd.DataFrame,
        sheet_name: str = 'Statistical_Results'
    ) -> None:
        """
        Add statistical test results sheet.

        Args:
            stats_df: DataFrame with statistical test results
            sheet_name: Sheet name
        """
        self.add_sheet(sheet_name, stats_df, index=False)

    def add_model_performance_sheet(
        self,
        performance_df: pd.DataFrame,
        sheet_name: str = 'Model_Performance'
    ) -> None:
        """
        Add model performance metrics sheet.

        Args:
            performance_df: DataFrame with model metrics
            sheet_name: Sheet name
        """
        self.add_sheet(sheet_name, performance_df, index=False)

    def write(
        self,
        auto_adjust_columns: bool = True,
        freeze_panes: Optional[tuple] = (1, 1)
    ) -> None:
        """
        Write all sheets to Excel file.

        Args:
            auto_adjust_columns: Auto-adjust column widths
            freeze_panes: Freeze panes position (row, col) or None
        """
        if not self.sheets:
            warnings.warn("No sheets to write")
            return

        # Create parent directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Try different Excel engines for compatibility
        engines_to_try = ['openpyxl', 'xlsxwriter']
        last_error = None

        for engine in engines_to_try:
            try:
                self._write_with_engine(engine, auto_adjust_columns, freeze_panes)
                return  # Success, exit
            except Exception as e:
                last_error = e
                if 'openpyxl' in str(e) or 'numpy' in str(e).lower():
                    warnings.warn(
                        f"Excel engine '{engine}' failed due to compatibility issue. "
                        f"Trying alternative..."
                    )
                    continue
                else:
                    # Different error, raise it
                    raise

        # If all engines failed, raise the last error
        raise ValueError(
            f"Failed to write Excel file with all available engines. "
            f"Last error: {last_error}\n\n"
            f"Solution: Upgrade openpyxl with: pip install --upgrade openpyxl>=3.1.0\n"
            f"Or install xlsxwriter: pip install xlsxwriter"
        )

    def _write_with_engine(
        self,
        engine: str,
        auto_adjust_columns: bool,
        freeze_panes: Optional[tuple]
    ) -> None:
        """Write Excel file with specified engine."""
        try:
            with pd.ExcelWriter(self.filepath, engine=engine) as writer:
                for sheet_name, sheet_info in self.sheets.items():
                    data = sheet_info['data']
                    include_index = sheet_info['index']

                    # Write sheet
                    data.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=include_index
                    )

                    # Apply formatting only for openpyxl (has worksheet object)
                    if engine == 'openpyxl':
                        # Get worksheet for formatting
                        worksheet = writer.sheets[sheet_name]

                        # Auto-adjust column widths
                        if auto_adjust_columns:
                            for column in worksheet.columns:
                                max_length = 0
                                column_letter = column[0].column_letter

                                for cell in column:
                                    try:
                                        if cell.value:
                                            max_length = max(max_length, len(str(cell.value)))
                                    except (AttributeError, TypeError):
                                        # Skip cells with unprintable values
                                        pass

                                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                                worksheet.column_dimensions[column_letter].width = adjusted_width

                        # Freeze panes
                        if freeze_panes:
                            row, col = freeze_panes
                            cell = worksheet.cell(row + 1, col + 1)
                            worksheet.freeze_panes = cell.coordinate

            print(f"Excel file written successfully: {self.filepath}")
            print(f"  Engine used: {engine}")
            print(f"  Sheets: {list(self.sheets.keys())}")

        except ImportError as e:
            raise ImportError(
                f"Excel engine '{engine}' not available. "
                f"Install with: pip install {engine}"
            ) from e

    def clear(self) -> None:
        """Clear all sheets."""
        self.sheets.clear()


class CSVExporter:
    """
    Export analysis results to CSV files.

    Simpler than Excel, but creates multiple CSV files instead of sheets.
    """

    def __init__(self, output_directory: str):
        """
        Initialize CSV exporter.

        Args:
            output_directory: Directory for output CSV files
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def export_features(
        self,
        features_df: pd.DataFrame,
        filename: str = 'features.csv'
    ) -> None:
        """
        Export features to CSV.

        Args:
            features_df: Features DataFrame
            filename: Output filename
        """
        filepath = self.output_directory / filename

        try:
            features_df.to_csv(filepath, index=True)
            print(f"Features exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export features: {e}")

    def export_statistical_results(
        self,
        stats_df: pd.DataFrame,
        filename: str = 'statistical_results.csv'
    ) -> None:
        """
        Export statistical results to CSV.

        Args:
            stats_df: Statistical results DataFrame
            filename: Output filename
        """
        filepath = self.output_directory / filename

        try:
            stats_df.to_csv(filepath, index=False)
            print(f"Statistical results exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export statistical results: {e}")

    def export_model_performance(
        self,
        performance_df: pd.DataFrame,
        filename: str = 'model_performance.csv'
    ) -> None:
        """
        Export model performance to CSV.

        Args:
            performance_df: Performance metrics DataFrame
            filename: Output filename
        """
        filepath = self.output_directory / filename

        try:
            performance_df.to_csv(filepath, index=False)
            print(f"Model performance exported to: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to export model performance: {e}")


class ResultsExporter:
    """
    High-level exporter that organizes all analysis results.

    Creates structured output:
    - Single Excel file with multiple sheets
    - OR organized directory with CSV files
    - Includes metadata and timestamps
    """

    def __init__(
        self,
        output_path: str,
        format: str = 'excel',
        include_timestamp: bool = True
    ):
        """
        Initialize results exporter.

        Args:
            output_path: Output file path (Excel) or directory (CSV)
            format: 'excel' or 'csv'
            include_timestamp: Add timestamp to filename
        """
        if format not in ['excel', 'csv']:
            raise ValueError(f"Format must be 'excel' or 'csv', got '{format}'")

        self.output_path = Path(output_path)
        self.format = format
        self.include_timestamp = include_timestamp

        # Add timestamp to filename if requested
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if format == 'excel':
                # For Excel, add timestamp before extension
                stem = self.output_path.stem
                suffix = self.output_path.suffix or '.xlsx'
                parent = self.output_path.parent
                self.output_path = parent / f"{stem}_{timestamp}{suffix}"
            else:
                # For CSV, create timestamped directory
                self.output_path = self.output_path / timestamp

        # Ensure proper extension for Excel
        if format == 'excel' and not str(self.output_path).endswith('.xlsx'):
            self.output_path = self.output_path.with_suffix('.xlsx')

    def export_complete_analysis(
        self,
        features: pd.DataFrame,
        statistical_results: pd.DataFrame,
        model_performance: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export complete analysis results.

        Args:
            features: Extracted features
            statistical_results: Statistical test results
            model_performance: Model performance metrics
            metadata: Optional analysis metadata

        Returns:
            Path to exported file(s)
        """
        if self.format == 'excel':
            exporter = ExcelExporter(str(self.output_path))

            # Add sheets
            exporter.add_features_sheet(features)
            exporter.add_statistical_results_sheet(statistical_results)
            exporter.add_model_performance_sheet(model_performance)

            # Add metadata sheet if provided
            if metadata:
                metadata_df = pd.DataFrame([metadata])
                exporter.add_sheet('Metadata', metadata_df, index=False)

            # Write
            exporter.write()

            return str(self.output_path)

        else:  # CSV format
            exporter = CSVExporter(str(self.output_path))

            # Export each component
            exporter.export_features(features)
            exporter.export_statistical_results(statistical_results)
            exporter.export_model_performance(model_performance)

            # Export metadata if provided
            if metadata:
                metadata_df = pd.DataFrame([metadata])
                metadata_path = self.output_path / 'metadata.csv'
                metadata_df.to_csv(metadata_path, index=False)
                print(f"Metadata exported to: {metadata_path}")

            return str(self.output_path)

    def export_with_labels(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        statistical_results: pd.DataFrame,
        model_performance: pd.DataFrame
    ) -> str:
        """
        Export analysis results including labels.

        Args:
            features: Features DataFrame
            labels: Labels DataFrame
            statistical_results: Statistical results
            model_performance: Model performance

        Returns:
            Path to exported file(s)
        """
        if self.format == 'excel':
            exporter = ExcelExporter(str(self.output_path))

            # Add sheets
            exporter.add_sheet('Labels', labels, index=False)
            exporter.add_features_sheet(features)
            exporter.add_statistical_results_sheet(statistical_results)
            exporter.add_model_performance_sheet(model_performance)

            # Write
            exporter.write()

            return str(self.output_path)

        else:  # CSV format
            exporter = CSVExporter(str(self.output_path))

            # Export labels
            labels_path = self.output_path / 'labels.csv'
            labels.to_csv(labels_path, index=False)
            print(f"Labels exported to: {labels_path}")

            # Export other components
            exporter.export_features(features)
            exporter.export_statistical_results(statistical_results)
            exporter.export_model_performance(model_performance)

            return str(self.output_path)