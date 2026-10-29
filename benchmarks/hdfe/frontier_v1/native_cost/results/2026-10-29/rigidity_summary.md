# Native cost study summary

Times are seconds: median [min, max] across passed calls. Counts retain their native units.

## Full call timing and accuracy

| Family | Backend | L | Diag | Call | Statuses | Total s | Max forward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rigidity_grid | fereg | 32 | off | cold | {"passed": 3} | 0.045683 [0.0447034, 0.0464676] | 1.14123e-12 |
| rigidity_grid | fereg | 32 | off | warm | {"passed": 3} | 0.0494266 [0.0435182, 0.0506326] | 1.14123e-12 |
| rigidity_grid | fereg | 32 | on | cold | {"passed": 3} | 0.0521388 [0.0495388, 0.0542736] | 1.14123e-12 |
| rigidity_grid | fereg | 32 | on | warm | {"passed": 3} | 0.052005 [0.0454181, 0.0684958] | 1.14123e-12 |
| rigidity_grid | fereg | 64 | off | cold | {"passed": 3} | 0.130462 [0.121204, 0.151021] | 7.4699e-14 |
| rigidity_grid | fereg | 64 | off | warm | {"passed": 3} | 0.0205525 [0.0161926, 0.0221349] | 7.35787e-14 |
| rigidity_grid | fereg | 64 | on | cold | {"passed": 3} | 0.134243 [0.128545, 0.135612] | 7.4699e-14 |
| rigidity_grid | fereg | 64 | on | warm | {"passed": 3} | 0.0152061 [0.0133568, 0.0160789] | 7.35787e-14 |
| rigidity_grid | fereg | 128 | off | cold | {"passed": 3} | 0.643229 [0.599978, 0.774009] | 1.38726e-13 |
| rigidity_grid | fereg | 128 | off | warm | {"passed": 3} | 0.0640965 [0.0634247, 0.0675568] | 1.27063e-13 |
| rigidity_grid | fereg | 128 | on | cold | {"passed": 3} | 0.703397 [0.617077, 0.710022] | 1.38726e-13 |
| rigidity_grid | fereg | 128 | on | warm | {"passed": 3} | 0.0682826 [0.0595206, 0.0716193] | 1.27063e-13 |
| rigidity_grid | fereg | 256 | off | cold | {"passed": 3} | 3.6898 [3.59867, 3.84931] | 1.68242e-12 |
| rigidity_grid | fereg | 256 | off | warm | {"passed": 3} | 0.269741 [0.261024, 0.278319] | 1.57741e-12 |
| rigidity_grid | fereg | 256 | on | cold | {"passed": 3} | 3.37715 [3.31569, 3.60368] | 1.68242e-12 |
| rigidity_grid | fereg | 256 | on | warm | {"passed": 3} | 0.275424 [0.255981, 0.290862] | 1.57741e-12 |
| rigidity_grid | native_auto | 32 | off | cold | {"passed": 3} | 0.106164 [0.0841206, 0.106388] | 3.76493e-12 |
| rigidity_grid | native_auto | 32 | off | warm | {"passed": 3} | 0.0250867 [0.0188785, 0.0298236] | 4.98181e-12 |
| rigidity_grid | native_auto | 32 | on | cold | {"passed": 3} | 0.0870818 [0.0803916, 0.0905757] | 3.76493e-12 |
| rigidity_grid | native_auto | 32 | on | warm | {"passed": 3} | 0.026955 [0.0220854, 0.0279484] | 4.98181e-12 |
| rigidity_grid | native_auto | 64 | off | cold | {"passed": 3} | 0.147645 [0.12968, 0.152978] | 4.35038e-12 |
| rigidity_grid | native_auto | 64 | off | warm | {"passed": 3} | 0.0926247 [0.0799519, 0.0935992] | 3.70607e-12 |
| rigidity_grid | native_auto | 64 | on | cold | {"passed": 3} | 0.133415 [0.127152, 0.15825] | 4.35038e-12 |
| rigidity_grid | native_auto | 64 | on | warm | {"passed": 3} | 0.0926687 [0.0752548, 0.109472] | 3.70607e-12 |
| rigidity_grid | native_auto | 128 | off | cold | {"passed": 3} | 0.636701 [0.564391, 0.648494] | 4.27006e-12 |
| rigidity_grid | native_auto | 128 | off | warm | {"passed": 3} | 0.430136 [0.424153, 0.447886] | 3.11819e-12 |
| rigidity_grid | native_auto | 128 | on | cold | {"passed": 3} | 0.636338 [0.630712, 0.65624] | 4.27006e-12 |
| rigidity_grid | native_auto | 128 | on | warm | {"passed": 3} | 0.422997 [0.422353, 0.43122] | 3.11819e-12 |
| rigidity_grid | native_auto | 256 | off | cold | {"passed": 3} | 2.71531 [2.71, 2.75441] | 5.02653e-12 |
| rigidity_grid | native_auto | 256 | off | warm | {"passed": 3} | 1.76182 [1.75399, 1.77487] | 3.56031e-12 |
| rigidity_grid | native_auto | 256 | on | cold | {"passed": 3} | 2.97611 [2.89991, 3.0289] | 5.02653e-12 |
| rigidity_grid | native_auto | 256 | on | warm | {"passed": 3} | 1.78136 [1.68573, 1.7952] | 3.56031e-12 |
| rigidity_grid | within | 32 | off | cold | {"passed": 3} | 0.0848216 [0.0723024, 0.090175] | 2.19504e-13 |
| rigidity_grid | within | 32 | off | warm | {"passed": 3} | 0.0174244 [0.0165923, 0.018805] | 2.23208e-13 |
| rigidity_grid | within | 32 | on | cold | {"passed": 3} | 0.0802127 [0.0765479, 0.100196] | 2.19504e-13 |
| rigidity_grid | within | 32 | on | warm | {"passed": 3} | 0.0184154 [0.0174357, 0.0188755] | 2.23208e-13 |
| rigidity_grid | within | 64 | off | cold | {"passed": 3} | 0.156046 [0.135268, 0.15668] | 1.99192e-13 |
| rigidity_grid | within | 64 | off | warm | {"passed": 3} | 0.0847513 [0.0729747, 0.0864701] | 1.85192e-13 |
| rigidity_grid | within | 64 | on | cold | {"passed": 3} | 0.1294 [0.12608, 0.148903] | 1.99192e-13 |
| rigidity_grid | within | 64 | on | warm | {"passed": 3} | 0.080002 [0.0728218, 0.0843038] | 1.85192e-13 |
| rigidity_grid | within | 128 | off | cold | {"passed": 3} | 0.633957 [0.589043, 0.722457] | 2.14785e-13 |
| rigidity_grid | within | 128 | off | warm | {"passed": 3} | 0.420421 [0.380667, 0.489028] | 1.54474e-13 |
| rigidity_grid | within | 128 | on | cold | {"passed": 3} | 0.649263 [0.557906, 0.744614] | 2.14785e-13 |
| rigidity_grid | within | 128 | on | warm | {"passed": 3} | 0.422343 [0.395114, 0.465373] | 1.54474e-13 |
| rigidity_grid | within | 256 | off | cold | {"passed": 3} | 2.89878 [2.80461, 2.90459] | 1.53054e-13 |
| rigidity_grid | within | 256 | off | warm | {"passed": 3} | 1.78231 [1.73637, 1.83796] | 2.27716e-13 |
| rigidity_grid | within | 256 | on | cold | {"passed": 3} | 2.8096 [2.6882, 2.87392] | 1.53054e-13 |
| rigidity_grid | within | 256 | on | warm | {"passed": 3} | 1.81857 [1.74621, 1.88534] | 2.27716e-13 |

## Within vector-call work (diagnostics on)

| Backend | L | Call | D apply | D adjoint | P apply | P adjoint | Deferred build s | Escalated | Local factors | Max factor dim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_auto | 32 | cold | 68 | 71 | 69 | 0 | 0.057549 [0.0560569, 0.0640573] | True | 3 | 2048 |
| native_auto | 32 | warm | 60 | 62 | 60 | 0 | 0 | True | 3 | 2048 |
| native_auto | 64 | cold | 69 | 72 | 70 | 0 | 0.0304014 [0.0283139, 0.0304439] | True | 3 | 8192 |
| native_auto | 64 | warm | 62 | 64 | 62 | 0 | 0 | True | 3 | 8192 |
| native_auto | 128 | cold | 71 | 74 | 72 | 0 | 0.152419 [0.141359, 0.176262] | True | 3 | 32768 |
| native_auto | 128 | warm | 64 | 66 | 64 | 0 | 0 | True | 3 | 32768 |
| native_auto | 256 | cold | 73 | 76 | 74 | 0 | 0.796388 [0.743825, 0.859772] | True | 3 | 131072 |
| native_auto | 256 | warm | 66 | 68 | 66 | 0 | 0 | True | 3 | 131072 |
| within | 32 | cold | 85 | 91 | 87 | 0 | 0.0532573 [0.0527516, 0.0754476] | True | 3 | 2048 |
| within | 32 | warm | 70 | 74 | 70 | 0 | 0 | True | 3 | 2048 |
| within | 64 | cold | 86 | 92 | 88 | 0 | 0.026659 [0.0242211, 0.0270324] | True | 3 | 8192 |
| within | 64 | warm | 72 | 76 | 72 | 0 | 0 | True | 3 | 8192 |
| within | 128 | cold | 88 | 94 | 90 | 0 | 0.142499 [0.121013, 0.157414] | True | 3 | 32768 |
| within | 128 | warm | 74 | 78 | 74 | 0 | 0 | True | 3 | 32768 |
| within | 256 | cold | 92 | 98 | 94 | 0 | 0.688789 [0.652097, 0.696163] | True | 3 | 131072 |
| within | 256 | warm | 76 | 80 | 76 | 0 | 0 | True | 3 | 131072 |

## Fereg batch work and factor state (diagnostics on)

| L | Call | H batches | P batches | Exact P | Joint allocated entries | CSR nnz | Build outcome | Joint build / outer total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | cold | 139 | 140 | 0 | — | — | — | 0 |
| 32 | warm | 139 | 140 | 0 | — | — | — | 0 |
| 64 | cold | 7 | 8 | 1 | 3168311 | 507904 | built | 0.758691 [0.73826, 0.795695] |
| 64 | warm | 0 | 1 | 1 | 3168311 | 507904 | built | 0 |
| 128 | cold | 7 | 8 | 1 | 16270722 | 2031616 | built | 0.798891 [0.777628, 0.809046] |
| 128 | warm | 0 | 1 | 1 | 16270722 | 2031616 | built | 0 |
| 256 | cold | 7 | 8 | 1 | 79707216 | 8126464 | built | 0.837808 [0.830841, 0.844732] |
| 256 | warm | 0 | 1 | 1 | 79707216 | 8126464 | built | 0 |

## Fereg inclusive phase times; columns must not be added

| L | Call | Joint build | Aggregation | Pattern | Assembly | Analyze incl. ordering | Numeric | Refactor | Refresh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | cold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 32 | warm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 64 | cold | 0.102887 [0.0948993, 0.106817] | 0.00137673 [0.00115695, 0.0015547] | 0.00117077 [0.000778319, 0.00128586] | 0.00937903 [0.00870127, 0.0112884] | 0.0228576 [0.02196, 0.02339] | 0.0676169 [0.0583709, 0.0720757] | 0 | 0 |
| 64 | warm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 128 | cold | 0.561937 [0.479857, 0.57444] | 0.00511365 [0.00470107, 0.00572884] | 0.00487349 [0.00345351, 0.00522118] | 0.0361469 [0.032031, 0.0372847] | 0.0942059 [0.0913559, 0.0961847] | 0.418016 [0.346112, 0.430945] | 0 | 0 |
| 128 | warm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256 | cold | 2.85278 [2.77791, 2.99408] | 0.0180704 [0.0165792, 0.0221535] | 0.0140088 [0.0127866, 0.0168414] | 0.148646 [0.126672, 0.156817] | 0.433683 [0.419495, 0.457537] | 2.25303 [2.13157, 2.37047] | 0 | 0 |
| 256 | warm | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Cold total / warm total, paired by worker

| Backend | L | diagnostics | Paired ratio | Excluded pairs |
| --- | --- | --- | --- | --- |
| fereg | 32 | False | 0.924259 [0.917739, 1.02724] | 0 |
| fereg | 32 | True | 1.00257 [0.792364, 1.09073] | 0 |
| fereg | 64 | False | 6.82278 [5.89725, 8.05687] | 0 |
| fereg | 64 | True | 8.91823 [7.99463, 10.0505] | 0 |
| fereg | 128 | False | 10.1416 [9.36054, 11.4572] | 0 |
| fereg | 128 | True | 9.91383 [9.03711, 11.8177] | 0 |
| fereg | 256 | False | 13.6791 [12.93, 14.747] | 0 |
| fereg | 256 | True | 12.3896 [12.2616, 12.9528] | 0 |
| native_auto | 32 | False | 3.55973 [3.35319, 5.63539] | 0 |
| native_auto | 32 | True | 3.36026 [3.1158, 3.64003] | 0 |
| native_auto | 64 | False | 1.62197 [1.59401, 1.63439] | 0 |
| native_auto | 64 | True | 1.44557 [1.4397, 1.68962] | 0 |
| native_auto | 128 | False | 1.42157 [1.33063, 1.50765] | 0 |
| native_auto | 128 | True | 1.49106 [1.47567, 1.55377] | 0 |
| native_auto | 256 | False | 1.54808 [1.52687, 1.56339] | 0 |
| native_auto | 256 | True | 1.68722 [1.62792, 1.76547] | 0 |
| within | 32 | False | 4.51058 [4.14949, 5.43475] | 0 |
| within | 32 | True | 4.39031 [4.24957, 5.44085] | 0 |
| within | 64 | False | 1.80462 [1.59606, 2.14704] | 0 |
| within | 64 | True | 1.73136 [1.61746, 1.76626] | 0 |
| within | 128 | False | 1.66538 [1.20452, 1.71842] | 0 |
| within | 128 | True | 1.60004 [1.32098, 1.64323] | 0 |
| within | 256 | False | 1.62642 [1.52593, 1.6728] | 0 |
| within | 256 | True | 1.52436 [1.47819, 1.60897] | 0 |

## Diagnostics on / off total, paired by replicate

| Backend | L | mode | Paired ratio | Excluded pairs |
| --- | --- | --- | --- | --- |
| fereg | 32 | cold | 1.16633 [1.0844, 1.16799] | 0 |
| fereg | 32 | warm | 1.19502 [0.9189, 1.3528] | 0 |
| fereg | 64 | cold | 1.03948 [0.888901, 1.06057] | 0 |
| fereg | 64 | warm | 0.78233 [0.603429, 0.939078] | 0 |
| fereg | 128 | cold | 1.0285 [0.908771, 1.10384] | 0 |
| fereg | 128 | warm | 1.06531 [0.881045, 1.1292] | 0 |
| fereg | 256 | cold | 0.938444 [0.861371, 0.976659] | 0 |
| fereg | 256 | warm | 0.989597 [0.980682, 1.0783] | 0 |
| native_auto | 32 | cold | 0.853168 [0.81853, 0.955671] | 0 |
| native_auto | 32 | warm | 0.903815 [0.880362, 1.48043] | 0 |
| native_auto | 64 | cold | 0.980508 [0.903621, 1.03446] | 0 |
| native_auto | 64 | warm | 1.00047 [0.941251, 1.16958] | 0 |
| native_auto | 128 | cold | 0.99943 [0.97258, 1.16274] | 0 |
| native_auto | 128 | warm | 0.983403 [0.962789, 0.995755] | 0 |
| native_auto | 256 | cold | 1.0982 [1.06798, 1.09965] | 0 |
| native_auto | 256 | warm | 1.01561 [0.94978, 1.01895] | 0 |
| within | 32 | cold | 0.945664 [0.848882, 1.38579] | 0 |
| within | 32 | warm | 1.05083 [1.00375, 1.05687] | 0 |
| within | 64 | cold | 0.932081 [0.829241, 0.950365] | 0 |
| within | 64 | warm | 0.925198 [0.859242, 1.15525] | 0 |
| within | 128 | cold | 0.947141 [0.898687, 1.17455] | 0 |
| within | 128 | warm | 0.939807 [0.863637, 1.22252] | 0 |
| within | 256 | cold | 0.967298 [0.958494, 0.991424] | 0 |
| within | 256 | warm | 1.00567 [0.98945, 1.05781] | 0 |

## Scope and repeat checks

- Worker/call counts: `{"call_statuses": {"passed": 144}, "recorded_calls": 144, "represented_call_outcomes": 144, "worker_cases": 72, "worker_statuses": {"passed": 72}}`.
- Factor entries are allocated values, including padding; symbolic flops are estimates.
- Snapshot constructor/cached-build times are metadata, not new costs on each warm call.
- Ratios describe these trials; diagnostics on/off does not measure the entire patch overhead.
- Expected repeat coverage, controls, native counter values and structural snapshots agree across observed repeats.
