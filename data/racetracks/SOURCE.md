# Source of the circuit data

The files in `tracks/`, `LICENSE` and `README.md` are copied unchanged from the racetrack database of the Institute of
Automotive Technology, Technical University of Munich: https://github.com/TUMFTM/racetrack-database, commit
e59595d1f3573b30d1ded6a08984935b957688e0 (18 September 2021), licensed under the GNU Lesser General Public License
version 3 (LICENSE). Each file holds a circuit's centre line (`x_m`, `y_m`, from OpenStreetMap GPS points, smoothed) and
the road's width to the right and to the left of every point (`w_tr_right_m`, `w_tr_left_m`, from satellite images),
about 5 m apart, in the circuit's racing direction from the start–finish line. EC Racing reads them in
`racing/circuits.py` (specification §4.7); the race lines of the database are not used.
