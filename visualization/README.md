# Visualization module

You can run the following commands to visulize loss and metric curves:

```bash
python visualization/visualize.py --checkpoint_dir /path/to/checkpoint_dir --output_dir /path/to/output_dir \
--metrics additional_metric1 additional_metric2
```

You can also create comparison visualizations for multiple runs:

```bash
python visualization/visualize_comparison.py --checkpoint_paths /path/to/checkpoint_dir1 /path/to/checkpoint_dir2 \
--labels "Run 1" "Run 2" --metrics additional_metric1 additional_metric2 --max_steps <optional_max_steps>
```
