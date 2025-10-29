# OpenEnv RFCs

This directory contains Requests for Comments (RFCs) for the OpenEnv framework. RFCs are used to propose, discuss, and document significant changes, new features, or architectural decisions.

## RFC Process

### 1. Proposing an RFC

When you want to propose a significant change or feature:

1. **Create a new RFC file** in this directory named `NNN-short-title.md` where `NNN` is the next available RFC number based on open RFCs (e.g., `002-render-api.md`)
2. **Use the RFC template** structure outlined below
3. **Submit for review** by submitting a PR
4. **Iterate based on feedback** until consensus is reached

### 2. RFC Lifecycle

RFCs go through several stages:

- **In Review**: Open for feedback and discussion
- **Accepted**: Consensus reached, ready for implementation
- **Implemented**: Changes have been merged
- **Rejected**: Proposal was not accepted (document why)
- **Superseded**: Replaced by a newer RFC

### 3. What Requires an RFC?

You should write an RFC for:

- **New core APIs** (e.g., new methods on `Environment` or `HTTPEnvClient`)
- **Breaking changes** to existing interfaces
- **Major architectural decisions** (e.g., communication protocol changes)
- **New abstractions** or design patterns

You generally don't need an RFC for:

- Bug fixes
- Documentation improvements
- Minor refactoring
- New example environments (unless they introduce new patterns)

## RFC Template

Each RFC should include the following sections:

### Required Sections

1. **Header**
   ```markdown
   # RFC: [Title]
   
   **Status**: [Draft|In Review|Accepted|Implemented|Rejected|Superseded]
   **Created**: [Date]
   **Authors**: [@username1, @username2]
   **RFC ID**: [NNN]
   ```

2. **Summary**
   - Brief 1-2 paragraph overview of the proposal
   - Should be clear enough for someone to understand the essence without reading the full RFC

3. **Motivation**
   - **Problem Statement**: What problem are you solving?
   - **Goals**: What are you trying to achieve?
   - Clear explanation of why this change is needed

4. **Design**
   - **Architecture Overview**: High-level view (diagrams encouraged)
   - **Core Abstractions**: Key interfaces, classes, or concepts
   - **Key Design Decisions**: For each significant decision:
     - **Chosen Approach**: What you're proposing
     - **Rationale**: Why this approach
     - **Trade-offs**: What are the pros/cons if any

5. **Examples**
   - Code examples showing how the feature would be used
   - Should demonstrate common use cases
   - Include both client and server perspectives where relevant

## Current RFCs

- [001-openenv-spec.md](./001-openenv-spec.md) - OpenEnv Framework Specification (baseline APIs, HTTP communication, Docker isolation)

## Questions?

For questions about the RFC process, reach out to the core team or open a discussion in the project repository.
