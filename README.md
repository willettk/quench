quench
======

Code to reduce data from the Galaxy Zoo: Quench project. 

To create the full collated data set, begin in Python:

```
import quench_collate as qc
listcoll = qc.collate_answers()
qc.write_fits(listcoll)
```

After writing out the FITS file, open it in TOPCAT and match it against the final list of sample and control galaxies provided by Laura

Match on SDSS objid against ```finalpush/control_091113.csv``` and ```finalpush/sample_091113.csv```. These files contain the SDSS metadata and serve as the master list for galaxies of the final sample.

Output as FITS and CSV files from TOPCAT.


