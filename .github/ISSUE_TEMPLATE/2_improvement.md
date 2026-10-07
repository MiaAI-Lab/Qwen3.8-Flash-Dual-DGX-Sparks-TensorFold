---
name: Improvement
about: Something in this kit already works, but could be faster, safer, clearer or better defaulted.
title: ""
labels: "enhancement"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you want to propose an improvement to something that already exists, fill
     out the template below. For something that does not exist at all, please use
     the Feature request template instead.
-->

## What could be better

<!--
     Describe the current behavior and why it is not ideal. Examples that fit this
     repo:
       * a default in scripts/config.sh that is wrong for most people
       * a memory setting that is too conservative or does not fit
       * a patch that no longer applies to a newer TensorFold commit
       * startup time (the weights load on both Sparks)
-->

## Proposed change

<!--
     What should it do instead? If this touches TensorFold's code, it belongs in a
     patch under patches/ (a git diff against TensorFold's source at the commit
     pinned in scripts/config.sh, applied with git apply by scripts/prepare.sh).
     Drafted replies must still equal serial ones; say whether the patch keeps the
     same bits.
-->

## Measured impact

<!--
     This kit lives or dies on measured numbers, so please include them where you
     can. The README records results in this shape:

       * prefill tok/s and time to first token at several prompt sizes, and decode
         tok/s for prose and code replies at 1 to 4 concurrent requests,
         measured with sparkDash (https://github.com/MiaAI-Lab/sparkDash)
       * each rank's free memory at idle and at its lowest under a long prompt
       * needle retrieval PASS/FAIL

     Before/after pairs are much more useful than a single number, and please say
     which configuration each number came from (PARALLEL, CONTEXT, DRAFTS) - they
     move a lot between settings, and so do the GPU clocks.

     If you are comparing two configurations, please use the same sample count for
     both, and alternate restarts: boot-to-boot noise on a Spark is a few percent.
-->

## Risk

<!--
     Does this change memory use or output? A setting that raises memory can make the
     server refuse the default 4 x 1,048,576-token configuration; a patch that changes
     replies (beyond a labelled, measured lossy option) is not acceptable. Say so here
     if it does.
-->
