# PathSnap dataset exploration

Source: `Student folder`

## train

- Trips: 80
- GPS rows (min/median/max): {'min': 361, 'median': 1529.0, 'max': 19313}
- Nodes (min/median/max): {'min': 299, 'median': 9608.0, 'max': 155003}
- Arcs (min/median/max): {'min': 649, 'median': 19526.5, 'max': 307392}
- Route rows (min/median/max): {'min': 59, 'median': 280.0, 'max': 2266}
- Missing-file trips: 0
- Invalid coordinate rows: 0
- Nonincreasing timestamp steps: 0
- Non-1 s timestamp steps: 7284; exact-1 s fraction (min/median/max): {'min': 0.5306553911205074, 'median': 0.9914387810304783, 'max': 1.0}
- Timestamp gaps >1 s / >5 s: 7284 / 1216
- Invalid graph references: 0
- Invalid route references: 0
- Route connected-step fraction (min/median/max): {'min': 0.9813084112149533, 'median': 1.0, 'max': 1.0}
- Weak components per graph (min/median/max): {'min': 1, 'median': 20.0, 'max': 421}

## student_test

- Trips: 10
- GPS rows (min/median/max): {'min': 262, 'median': 1420.0, 'max': 3958}
- Nodes (min/median/max): {'min': 1770, 'median': 6596.5, 'max': 26635}
- Arcs (min/median/max): {'min': 3577, 'median': 13145.0, 'max': 56027}
- Route rows (min/median/max): {'min': 95, 'median': 250.0, 'max': 1323}
- Missing-file trips: 0
- Invalid coordinate rows: 0
- Nonincreasing timestamp steps: 0
- Non-1 s timestamp steps: 264; exact-1 s fraction (min/median/max): {'min': 0.8612794612794613, 'median': 0.9993230606354611, 'max': 1.0}
- Timestamp gaps >1 s / >5 s: 264 / 44
- Invalid graph references: 0
- Invalid route references: 0
- Route connected-step fraction (min/median/max): {'min': 0.9973190348525469, 'median': 1.0, 'max': 1.0}
- Weak components per graph (min/median/max): {'min': 5, 'median': 11.0, 'max': 100}
