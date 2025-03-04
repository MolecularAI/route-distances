route-distances documentation
===========================

route-distances contains routines and tools to compute distances between synthesis routes and to cluster them.

This is a project mostly targeting developers and researchers, not so much end-users.

There is one command line tool to process AiZynthFinder output:

.. code-block:: bash

    cluster_aizynth_output --files finder_output1.hdf5 finder_output2.hdf5 --output finder_distances.hdf5


This will add a column to table in the merged ``hdf5`` file called ``distance_matrix`` with the tree edit distances, and
another column called ``distances_time`` with the timings of the calculations.

To cluster the routes as well add the ``--ncluster`` flag

.. code-block:: bash

    cluster_aizynth_output --files finder_output1.hdf5 finder_output2.hdf5 --output finder_distances.hdf5 --nclusters 0

Giving 0 as the number of clusters will trigger automatic optimization of the number of clusters.
Two columns will be added to the table: ``cluster_labels`` and ``cluster_time`` holding the
cluster labels for each route and the timings of the calculation, respectively.




.. toctree::
    :hidden:

    python
    route_distances
