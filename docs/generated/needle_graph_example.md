# Generated Needle Graph Example

## Collapsed RLM View

This is what the run looks like if recursive calls collapse to strings:

```text
root
  call_llm("scan first third")  -> not found
  call_llm("scan middle third") -> decoy, no code
  call_llm("scan final third")  -> candidate code 84721
  call_llm("verify candidate")  -> 84721 matches the requested needle
  final answer                  -> 84721
```

## Sequence View

This is the same run as calls and returns:

```mermaid
sequenceDiagram
    participant root as root
    participant root_chunk_0 as root.chunk_0
    participant root_chunk_1 as root.chunk_1
    participant root_chunk_2 as root.chunk_2
    participant root_chunk_2_candidate_a as root.chunk_2.candidate_a
    participant root_chunk_2_candidate_b as root.chunk_2.candidate_b
    participant root_verify as root.verify
    root->>+root_chunk_0: launch_subagent Scan first third for the hidden secret code.
    root->>+root_chunk_1: launch_subagent Scan middle third for the hidden secret code.
    root->>+root_chunk_2: launch_subagent Scan final third for the hidden secret code.
    root_chunk_0-->>-root: not found
    root_chunk_1-->>-root: decoy, no code
    root_chunk_2->>+root_chunk_2_candidate_a: launch_subagent Inspect candidate window A.
    root_chunk_2->>+root_chunk_2_candidate_b: launch_subagent Inspect candidate window B.
    root_chunk_2_candidate_a-->>-root_chunk_2: decoy: the code is not 12345
    root_chunk_2_candidate_b-->>-root_chunk_2: needle: the secret code is 84721
    root_chunk_2-->>-root: candidate code 84721
    root->>+root_verify: launch_subagent Verify candidate code 84721 against the origi...
    root_verify-->>-root: 84721 matches the requested needle
    root-->>root: done 84721
```

## Steppable Graph Snapshots

### 1. Root parks after spawning parallel children

```mermaid
flowchart TD
    n_775aca4fec50["root<br/><i>query</i>"]:::query
    n_6efe0da341dc["root<br/><i>action</i>"]:::action
    n_f254e1d0b615["root<br/><i>supervising</i>"]:::sup
    n_4c43503fd1d5["root.chunk_0<br/><i>query</i>"]:::query
    n_f8c79721330a["root.chunk_1<br/><i>query</i>"]:::query
    n_b819fe52c01c["root.chunk_2<br/><i>query</i>"]:::query
    n_775aca4fec50 -->|flows_to| n_6efe0da341dc
    n_6efe0da341dc -->|flows_to| n_f254e1d0b615
    n_6efe0da341dc -->|spawns| n_4c43503fd1d5
    n_6efe0da341dc -->|spawns| n_f8c79721330a
    n_6efe0da341dc -->|spawns| n_b819fe52c01c
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 2. First children finish while chunk_2 keeps working

```mermaid
flowchart TD
    n_775aca4fec50["root<br/><i>query</i>"]:::query
    n_6efe0da341dc["root<br/><i>action</i>"]:::action
    n_f254e1d0b615["root<br/><i>supervising</i>"]:::sup
    n_4c43503fd1d5["root.chunk_0<br/><i>query</i>"]:::query
    n_cd0c9cab5c4a["root.chunk_0<br/><i>action</i>"]:::action
    n_01b877b58700["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_f8c79721330a["root.chunk_1<br/><i>query</i>"]:::query
    n_544df024ea32["root.chunk_1<br/><i>action</i>"]:::action
    n_f08dfefc174a["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_b819fe52c01c["root.chunk_2<br/><i>query</i>"]:::query
    n_138a1299b54c["root.chunk_2<br/><i>action</i>"]:::action
    n_96ab3088a1be["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_38891fef3ec4["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_08ceac6039f2["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_775aca4fec50 -->|flows_to| n_6efe0da341dc
    n_6efe0da341dc -->|flows_to| n_f254e1d0b615
    n_4c43503fd1d5 -->|flows_to| n_cd0c9cab5c4a
    n_cd0c9cab5c4a -->|flows_to| n_01b877b58700
    n_f8c79721330a -->|flows_to| n_544df024ea32
    n_544df024ea32 -->|flows_to| n_f08dfefc174a
    n_b819fe52c01c -->|flows_to| n_138a1299b54c
    n_138a1299b54c -->|flows_to| n_96ab3088a1be
    n_6efe0da341dc -->|spawns| n_4c43503fd1d5
    n_6efe0da341dc -->|spawns| n_f8c79721330a
    n_6efe0da341dc -->|spawns| n_b819fe52c01c
    n_138a1299b54c -->|spawns| n_38891fef3ec4
    n_138a1299b54c -->|spawns| n_08ceac6039f2
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 3. chunk_2 resumes from candidate readers

```mermaid
flowchart TD
    n_775aca4fec50["root<br/><i>query</i>"]:::query
    n_6efe0da341dc["root<br/><i>action</i>"]:::action
    n_f254e1d0b615["root<br/><i>supervising</i>"]:::sup
    n_4c43503fd1d5["root.chunk_0<br/><i>query</i>"]:::query
    n_cd0c9cab5c4a["root.chunk_0<br/><i>action</i>"]:::action
    n_01b877b58700["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_f8c79721330a["root.chunk_1<br/><i>query</i>"]:::query
    n_544df024ea32["root.chunk_1<br/><i>action</i>"]:::action
    n_f08dfefc174a["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_b819fe52c01c["root.chunk_2<br/><i>query</i>"]:::query
    n_138a1299b54c["root.chunk_2<br/><i>action</i>"]:::action
    n_96ab3088a1be["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_298256982916["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    n_38891fef3ec4["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_3edb3213a745["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_282a418cdaa0["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_08ceac6039f2["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_ccc94f357f2f["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_da0f2a2004dc["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_775aca4fec50 -->|flows_to| n_6efe0da341dc
    n_6efe0da341dc -->|flows_to| n_f254e1d0b615
    n_4c43503fd1d5 -->|flows_to| n_cd0c9cab5c4a
    n_cd0c9cab5c4a -->|flows_to| n_01b877b58700
    n_f8c79721330a -->|flows_to| n_544df024ea32
    n_544df024ea32 -->|flows_to| n_f08dfefc174a
    n_b819fe52c01c -->|flows_to| n_138a1299b54c
    n_138a1299b54c -->|flows_to| n_96ab3088a1be
    n_96ab3088a1be -->|flows_to| n_298256982916
    n_38891fef3ec4 -->|flows_to| n_3edb3213a745
    n_3edb3213a745 -->|flows_to| n_282a418cdaa0
    n_08ceac6039f2 -->|flows_to| n_ccc94f357f2f
    n_ccc94f357f2f -->|flows_to| n_da0f2a2004dc
    n_6efe0da341dc -->|spawns| n_4c43503fd1d5
    n_6efe0da341dc -->|spawns| n_f8c79721330a
    n_6efe0da341dc -->|spawns| n_b819fe52c01c
    n_138a1299b54c -->|spawns| n_38891fef3ec4
    n_138a1299b54c -->|spawns| n_08ceac6039f2
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```

### 4. Root resumes and returns the answer

```mermaid
flowchart TD
    n_775aca4fec50["root<br/><i>query</i>"]:::query
    n_6efe0da341dc["root<br/><i>action</i>"]:::action
    n_f254e1d0b615["root<br/><i>supervising</i>"]:::sup
    n_04b6b0d066f4["root<br/><i>resume</i>"]:::resume
    n_a56ca26e1960["root<br/><i>action</i>"]:::action
    n_be71e3d850a9["root<br/><i>supervising</i>"]:::sup
    n_402349c90cf4["root<br/><i>result</i><br/>84721"]:::result
    n_4c43503fd1d5["root.chunk_0<br/><i>query</i>"]:::query
    n_cd0c9cab5c4a["root.chunk_0<br/><i>action</i>"]:::action
    n_01b877b58700["root.chunk_0<br/><i>result</i><br/>not found"]:::result
    n_f8c79721330a["root.chunk_1<br/><i>query</i>"]:::query
    n_544df024ea32["root.chunk_1<br/><i>action</i>"]:::action
    n_f08dfefc174a["root.chunk_1<br/><i>result</i><br/>decoy, no code"]:::result
    n_b819fe52c01c["root.chunk_2<br/><i>query</i>"]:::query
    n_138a1299b54c["root.chunk_2<br/><i>action</i>"]:::action
    n_96ab3088a1be["root.chunk_2<br/><i>supervising</i>"]:::sup
    n_298256982916["root.chunk_2<br/><i>result</i><br/>candidate code 84721"]:::result
    n_38891fef3ec4["root.chunk_2.candidate_a<br/><i>query</i>"]:::query
    n_3edb3213a745["root.chunk_2.candidate_a<br/><i>action</i>"]:::action
    n_282a418cdaa0["root.chunk_2.candidate_a<br/><i>result</i><br/>decoy: the code is not 12345"]:::result
    n_08ceac6039f2["root.chunk_2.candidate_b<br/><i>query</i>"]:::query
    n_ccc94f357f2f["root.chunk_2.candidate_b<br/><i>action</i>"]:::action
    n_da0f2a2004dc["root.chunk_2.candidate_b<br/><i>result</i><br/>needle: the secret code is 84721"]:::result
    n_766171592332["root.verify<br/><i>query</i>"]:::query
    n_af2cdcb45d4e["root.verify<br/><i>action</i>"]:::action
    n_a062a9ff68bf["root.verify<br/><i>result</i><br/>84721 matches the requested needle"]:::result
    n_775aca4fec50 -->|flows_to| n_6efe0da341dc
    n_6efe0da341dc -->|flows_to| n_f254e1d0b615
    n_f254e1d0b615 -->|flows_to| n_04b6b0d066f4
    n_04b6b0d066f4 -->|flows_to| n_a56ca26e1960
    n_a56ca26e1960 -->|flows_to| n_be71e3d850a9
    n_be71e3d850a9 -->|flows_to| n_402349c90cf4
    n_4c43503fd1d5 -->|flows_to| n_cd0c9cab5c4a
    n_cd0c9cab5c4a -->|flows_to| n_01b877b58700
    n_f8c79721330a -->|flows_to| n_544df024ea32
    n_544df024ea32 -->|flows_to| n_f08dfefc174a
    n_b819fe52c01c -->|flows_to| n_138a1299b54c
    n_138a1299b54c -->|flows_to| n_96ab3088a1be
    n_96ab3088a1be -->|flows_to| n_298256982916
    n_38891fef3ec4 -->|flows_to| n_3edb3213a745
    n_3edb3213a745 -->|flows_to| n_282a418cdaa0
    n_08ceac6039f2 -->|flows_to| n_ccc94f357f2f
    n_ccc94f357f2f -->|flows_to| n_da0f2a2004dc
    n_766171592332 -->|flows_to| n_af2cdcb45d4e
    n_af2cdcb45d4e -->|flows_to| n_a062a9ff68bf
    n_6efe0da341dc -->|spawns| n_4c43503fd1d5
    n_6efe0da341dc -->|spawns| n_f8c79721330a
    n_6efe0da341dc -->|spawns| n_b819fe52c01c
    n_138a1299b54c -->|spawns| n_38891fef3ec4
    n_138a1299b54c -->|spawns| n_08ceac6039f2
    n_a56ca26e1960 -->|spawns| n_766171592332
    classDef query    fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef obs      fill:#1f6feb22,stroke:#58a6ff,color:#c9d1d9;
    classDef action   fill:#d2992222,stroke:#d29922,color:#c9d1d9;
    classDef sup      fill:#bc8cff22,stroke:#bc8cff,color:#c9d1d9;
    classDef resume   fill:#7ee78722,stroke:#7ee787,color:#c9d1d9;
    classDef err      fill:#f8514922,stroke:#f85149,color:#c9d1d9;
    classDef result   fill:#3fb95022,stroke:#3fb950,color:#c9d1d9;
```
