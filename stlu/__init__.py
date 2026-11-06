"""Shared, ComfyUI-independent helpers for the SeeThrough tile pipeline.

Modules here must stay importable without torch, folder_paths or a running
PromptServer: the node layer passes paths and logger callbacks in.
"""
