package io.dlminer.knn;

import java.util.Comparator;

public class ExampleComparator implements Comparator<DatasetEntry> {

    private double[] distances;

    public ExampleComparator(double[] distances) {
        super();
        this.distances = distances;
    }

    @Override
    public int compare(DatasetEntry o1, DatasetEntry o2) {
        return Double.compare(distances[o1.id], distances[o2.id]);
    }

}