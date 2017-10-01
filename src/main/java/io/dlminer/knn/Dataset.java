package io.dlminer.knn;

import java.util.ArrayList;
import java.util.Arrays;

public class Dataset extends ArrayList<DatasetEntry>{

    public static final int MAX_SIZE = 100;

    private static final long serialVersionUID = 1L;

    public static final double[] classBounds = new double[]{
            1, 10, 100, 1000
    };

    public Dataset() {
        super();
    }

    public Dataset(int[] sizes) {
        super();
        for (int s=0; s<sizes.length; s++) {
            DatasetEntry en = new DatasetEntry();
            en.id = s;
            en.vector = new Double[1];
            en.vector[0] = new Double(s);
            this.add(en);
        }
    }

    public void setLabels() {
        for (DatasetEntry ex: this) {
            ex.label = estimateLabel(ex.time);
        }
    }

    public void rescale() {
        int nfeats = this.get(0).vector.length;
        double[] means = new double[nfeats];
        double[] devs = new double[nfeats];
        Arrays.fill(means, 0.0);
        Arrays.fill(devs, 0.0);
        // calculate the mean for each feature
        for (DatasetEntry ex: this) {
            for (int i=0; i<nfeats; i++) {
                means[i] += ex.vector[i];
            }
        }
        for (int i=0; i<nfeats; i++) {
            means[i] /= this.size();
        }
        // calculate the standard deviation for each feature
        for (DatasetEntry ex: this) {
            for (int i=0; i<nfeats; i++) {
                devs[i] += (ex.vector[i]-means[i])*(ex.vector[i]-means[i]);
            }
        }
        for (int i=0; i<nfeats; i++) {
            devs[i] = Math.sqrt(devs[i]/this.size());
        }
        // subtract the mean from each example and divide by the deviation
        for (DatasetEntry ex: this) {
            for (int i=0; i<nfeats; i++) {
                ex.vector[i] = (devs[i]!=0) ? (ex.vector[i] - means[i])/devs[i] : (ex.vector[i] - means[i]);
            }
        }
    }

    public void setTimes(double[] times) {
        for (int i=0; i<times.length; i++) {
            DatasetEntry en = this.get(i);
            en.time = times[i];
        }
    }

    public static int estimateLabel(double time) {
        int label = -1;
        if (time<classBounds[0]) {
            label = 0;
        } else if (time<classBounds[1]) {
            label = 1;
        } else if (time<classBounds[2]) {
            label = 2;
        } else if (time<classBounds[3]) {
            label = 3;
        } else {
            label = 4;
        }
        return label;
    }


    public void extend() {
        if (this.size()<MAX_SIZE) {
            DatasetEntry last = this.get(this.size()-1);
            for (int i=this.size(); i<MAX_SIZE; i++) {
                this.add(last);
            }
        }
    }



    public DatasetEntry getById(int id) {
        for (DatasetEntry en : this) {
            if (en.id == id) {
                return en;
            }
        }
        return null;
    }

    public void updateVectors(double[][] data) {
        for (int i=0; i<data.length; i++) {
            Double[] vec = new Double[data[0].length];
            for (int j=0; j<vec.length; j++) {
                vec[j] = data[i][j];
            }
            this.get(i).vector = vec;
        }
    }

    public void print() {
        System.out.println("Dataset:");
        for (DatasetEntry en : this) {
            for (int i=0; i<en.vector.length; i++) {
                System.out.print(en.vector[i]+", ");
            }
            System.out.println();
        }
    }


    public boolean isTrivial() {
        for (DatasetEntry en : this) {
            if (!en.isTrivial()) {
                return false;
            }
        }
        return true;
    }
}