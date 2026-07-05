#!/bin/bash

################################################################################
# Multi-Resolution Throughput Testing Script
# Tests Vim, VMamba, and MambaVision across multiple image resolutions
# Resolutions: 224, 256, 384, 512, 768, 1024
################################################################################

set -e  # Exit on error

# ============================================================================
# Setup Phase
# ============================================================================

# Define resolutions to test
RESOLUTIONS=(224 256 384 512 768 1024)

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Initialize log files
VIM_LOG="$RESULTS_DIR/vim_throughput.log"
VMAMBA_LOG="$RESULTS_DIR/vmamba_throughput.log"
MAMBAVISION_LOG="$RESULTS_DIR/mambavision_throughput.log"
SUMMARY_REPORT="$RESULTS_DIR/summary_report.txt"

# Get workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Initialize conda (run once for entire script)
eval "$(conda shell.bash hook)"

# Create header for each log
echo "========================================" > "$VIM_LOG"
echo "Vim Throughput Tests" >> "$VIM_LOG"
echo "Started: $(date)" >> "$VIM_LOG"
echo "========================================" >> "$VIM_LOG"
echo "" >> "$VIM_LOG"

echo "========================================" > "$VMAMBA_LOG"
echo "VMamba Throughput Tests" >> "$VMAMBA_LOG"
echo "Started: $(date)" >> "$VMAMBA_LOG"
echo "========================================" >> "$VMAMBA_LOG"
echo "" >> "$VMAMBA_LOG"

echo "========================================" > "$MAMBAVISION_LOG"
echo "MambaVision Throughput Tests" >> "$MAMBAVISION_LOG"
echo "Started: $(date)" >> "$MAMBAVISION_LOG"
echo "========================================" >> "$MAMBAVISION_LOG"
echo "" >> "$MAMBAVISION_LOG"

# Initialize summary report
{
    echo "========================================"
    echo "Throughput Testing Summary Report"
    echo "========================================"
    echo "Generated: $(date)"
    echo "Workspace: $WORKSPACE_ROOT"
    echo "Results Directory: $RESULTS_DIR"
    echo ""
} > "$SUMMARY_REPORT"

echo "✓ Results directory created: $RESULTS_DIR"
echo "✓ Log files initialized"
echo ""

# ============================================================================
# Phase 1: Vim Throughput Tests
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  Vim Throughput Testing                        ║"
echo "║        (Model: vim_tiny, Batch Size: 16)                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Phase 1: Vim Throughput Tests" >> "$SUMMARY_REPORT"
echo "=============================" >> "$SUMMARY_REPORT"
echo "Model: vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2" >> "$SUMMARY_REPORT"
echo "Batch Size: 16" >> "$SUMMARY_REPORT"
echo "Checkpoint: vim_t_midclstok_76p1acc.pth" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

conda activate vim

for resolution in "${RESOLUTIONS[@]}"; do
    echo "  Testing Vim at ${resolution}px..."
    
    # Log with timestamp
    {
        echo "─────────────────────────────────────"
        echo "Resolution: ${resolution}px"
        echo "Timestamp: $(date)"
        echo "─────────────────────────────────────"
    } >> "$VIM_LOG"
    
    # Run throughput test and capture output
    cd "$WORKSPACE_ROOT"
    python Vim/vim/measure_throughput.py \
        --resolution "$resolution" \
        --bs 16 \
        --model "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2" \
        --ckpt "vim/vim_t_midclstok_76p1acc.pth" \
        >> "$VIM_LOG" 2>&1 || {
        echo "    ✗ Error testing Vim at ${resolution}px"
        echo "Error at ${resolution}px: See log for details" >> "$SUMMARY_REPORT"
        continue
    }
    
    echo "    ✓ Completed"
done

echo "✓ Vim throughput tests completed"
echo "" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

# ============================================================================
# Phase 2: VMamba Throughput Tests
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                 VMamba Throughput Testing                      ║"
echo "║        (Model: vmamba_tiny_s1l8, Batch Size: 16)               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Phase 2: VMamba Throughput Tests" >> "$SUMMARY_REPORT"
echo "================================" >> "$SUMMARY_REPORT"
echo "Model: vmamba_tiny_s1l8" >> "$SUMMARY_REPORT"
echo "Batch Size: 16" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

conda activate vmamba

for resolution in "${RESOLUTIONS[@]}"; do
    echo "  Testing VMamba at ${resolution}px..."
    
    # Log with timestamp
    {
        echo "─────────────────────────────────────"
        echo "Resolution: ${resolution}px"
        echo "Timestamp: $(date)"
        echo "─────────────────────────────────────"
    } >> "$VMAMBA_LOG"
    
    # Run throughput test and capture output
    cd "$WORKSPACE_ROOT"
    python VMamba/throughput_analysis.py \
        --resolution "$resolution" \
        --bs 16 \
        --model "vmamba_tiny_s1l8" \
        >> "$VMAMBA_LOG" 2>&1 || {
        echo "    ✗ Error testing VMamba at ${resolution}px"
        echo "Error at ${resolution}px: See log for details" >> "$SUMMARY_REPORT"
        continue
    }
    
    echo "    ✓ Completed"
done

echo "✓ VMamba throughput tests completed"
echo "" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

# ============================================================================
# Phase 3: MambaVision Throughput Tests
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              MambaVision Throughput Testing                    ║"
echo "║        (Model: mamba_vision_T, Batch Size: 128)                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "Phase 3: MambaVision Throughput Tests" >> "$SUMMARY_REPORT"
echo "=====================================" >> "$SUMMARY_REPORT"
echo "Model: mamba_vision_T" >> "$SUMMARY_REPORT"
echo "Batch Size: 128" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

conda activate mambavision

for resolution in "${RESOLUTIONS[@]}"; do
    echo "  Testing MambaVision at ${resolution}px..."
    
    # Log with timestamp
    {
        echo "─────────────────────────────────────"
        echo "Resolution: ${resolution}px"
        echo "Timestamp: $(date)"
        echo "─────────────────────────────────────"
    } >> "$MAMBAVISION_LOG"
    
    # Run throughput test and capture output
    cd "$WORKSPACE_ROOT"
    python MambaVision/mambavision/throughput_measure.py \
        --resolution "$resolution" \
        --bs 128 \
        --model "mamba_vision_T" \
        >> "$MAMBAVISION_LOG" 2>&1 || {
        echo "    ✗ Error testing MambaVision at ${resolution}px"
        echo "Error at ${resolution}px: See log for details" >> "$SUMMARY_REPORT"
        continue
    }
    
    echo "    ✓ Completed"
done

echo "✓ MambaVision throughput tests completed"
echo "" >> "$SUMMARY_REPORT"

# ============================================================================
# Phase 4: Generate Summary Report
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   Generating Summary Report                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Extract throughput results from logs
echo "" >> "$SUMMARY_REPORT"
echo "Detailed Results:" >> "$SUMMARY_REPORT"
echo "=================" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

echo "Vim Results:" >> "$SUMMARY_REPORT"
echo "────────────" >> "$SUMMARY_REPORT"
grep -A2 "Resolution:" "$VIM_LOG" | grep -E "(Resolution:|Throughput)" >> "$SUMMARY_REPORT" || echo "No throughput data found" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

echo "VMamba Results:" >> "$SUMMARY_REPORT"
echo "───────────────" >> "$SUMMARY_REPORT"
grep -A2 "Resolution:" "$VMAMBA_LOG" | grep -E "(Resolution:|Throughput)" >> "$SUMMARY_REPORT" || echo "No throughput data found" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

echo "MambaVision Results:" >> "$SUMMARY_REPORT"
echo "───────────────────" >> "$SUMMARY_REPORT"
grep -A2 "Resolution:" "$MAMBAVISION_LOG" | grep -E "(Resolution:|Throughput)" >> "$SUMMARY_REPORT" || echo "No throughput data found" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

# Add completion info
{
    echo "========================================"
    echo "Test Completion: $(date)"
    echo "========================================"
} >> "$SUMMARY_REPORT"

# Display summary to console
echo "✓ Summary report generated"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                         TESTING COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Results Directory: $RESULTS_DIR"
echo ""
echo "Generated Files:"
echo "  • $VIM_LOG"
echo "  • $VMAMBA_LOG"
echo "  • $MAMBAVISION_LOG"
echo "  • $SUMMARY_REPORT"
echo ""
echo "To view the summary report:"
echo "  cat $SUMMARY_REPORT"
echo ""
