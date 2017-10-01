package io.dlminer.knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;

public class KNNClassifier {

    private Dataset dataset;

    private ArrayList<Integer> trainset;
    private ArrayList<Integer> testset;
    private ArrayList<Integer> holdout;

    private int k;

    private double[] distances;
    private PriorityQueue<DatasetEntry> queue;

    public KNNClassifier(Dataset dataset, int k) {
        super();
        this.dataset = dataset;
        this.k = k;
        init();
    }

    private void init() {
        dataset.setLabels();
        int lastId = dataset.get(dataset.size()-1).id;
        distances = new double[lastId+1];
        resetDistances();
        queue = new PriorityQueue<DatasetEntry>(k, new ExampleComparator(distances));
        trainset = new ArrayList<Integer>();
        testset = new ArrayList<Integer>();
        holdout = new ArrayList<Integer>();
    }

    public int classify(DatasetEntry ex, int[] fsel) {
        queue.clear();
        resetDistances();
        // recalculate distances and update the queue
        for (int i=0; i<trainset.size(); i++) {
            DatasetEntry en = dataset.get(trainset.get(i));
            distances[en.id] = distance(ex, en, fsel);
        }
        for (int i=0; i<trainset.size(); i++) {
            DatasetEntry en = dataset.get(trainset.get(i));
            queue.add(en);
        }
        // collect statistics
        int[] hist = new int[Dataset.classBounds.length+1];
        for (int i=0; i<k; i++) {
            DatasetEntry en = queue.poll();
            hist[en.label]++;
        }
        // find the most common label
        int max = -1;
        int label = -1;
        for (int i=0; i<hist.length; i++) {
            if (hist[i]>max) {
                max = hist[i];
                label = i;
            }
        }

        return label;
    }



    private double computeError(int[] fsel) {
        double error = 0;
        for (int i=0; i<testset.size(); i++) {
            DatasetEntry ex = dataset.get(testset.get(i));
            int label = classify(ex, fsel);
            if (label != ex.label) {
                error++;
            }
        }
        return error/testset.size();
    }

    private double holdoutError(int[] fsel) {
        double error = 0;
        for (int i=0; i<holdout.size(); i++) {
            DatasetEntry ex = dataset.get(holdout.get(i));
            int label = classify(ex, fsel);
            if (label != ex.label) {
                error++;
            }
        }
        return error/testset.size();
    }

    private double computePrediction(int[] fsel) {
        double error = 0;
        for (int i=0; i<testset.size(); i++) {
            DatasetEntry ex = dataset.get(testset.get(i));
            double time = predict(ex, fsel);
            error += (time - ex.time)*(time - ex.time);
        }
        return Math.sqrt(error/testset.size());
    }

    private double holdoutPrediction(int[] fsel) {
        double error = 0;
        for (int i=0; i<holdout.size(); i++) {
            DatasetEntry ex = dataset.get(holdout.get(i));
            double time = predict(ex, fsel);
            error += (time - ex.time)*(time - ex.time);
        }
        return Math.sqrt(error/holdout.size());
    }

    private double predict(DatasetEntry ex, int[] fsel) {
        queue.clear();
        resetDistances();
        // recalculate distances and update the queue
        for (int i=0; i<trainset.size(); i++) {
            DatasetEntry en = dataset.get(trainset.get(i));
            distances[en.id] = distance(ex, en, fsel);
        }
        for (int i=0; i<trainset.size(); i++) {
            DatasetEntry en = dataset.get(trainset.get(i));
            queue.add(en);
        }
        // compute weights
//		double[] weights = new double[k];
        double[] times = new double[k];
        for (int i=0; i<k; i++) {
            DatasetEntry en = queue.poll();
//			weights[i] = 1/distances[en.id];
            times[i] = en.time;
        }
//		normalizeWeights(weights);
        // collect values
        double time = 0;
        for (int i=0; i<k; i++) {
            time += times[i]/k;//*weights[i];
        }
        return time;
    }

    private static void normalizeWeights(double[] weights) {
        double sum = 0;
        for (int i=0; i<weights.length; i++) {
            sum += weights[i];
        }
        for (int i=0; i<weights.length; i++) {
            weights[i] /= sum;
        }
    }

    private double distance(DatasetEntry ex1, DatasetEntry ex2, int[] fsel) {
        double dist = 0;
        Double[] vec1 = ex1.vector;
        Double[] vec2 = ex2.vector;
        for (int i=0; i<vec2.length; i++) {
            // Euclidian
            if (fsel[i] == 1) {
                dist += (vec1[i] - vec2[i])*(vec1[i] - vec2[i]);
            }
        }
        return Math.sqrt(dist);
    }

    private void resetDistances() {
        Arrays.fill(distances, Double.MAX_VALUE);
    }

    public void crossValidation(int[] fsel) {
        int repeats = 1000;
        int folds = 10;

        double[] errors = new double[folds*repeats];
        for (int r=0; r<repeats; r++) {
            ArrayList<Integer> ids = new ArrayList<Integer>();
            for (int i=0; i<dataset.size(); i++) {
                ids.add(i);
            }
            Collections.shuffle(ids);

            int step = ids.size()/folds;

            for (int i=0; i<folds; i++) {
                trainset.clear();
                testset.clear();
                // bounds
                int lbound = i*step;
                int rbound = (i == folds-1) ? ids.size()-1 : (i+1)*step-1;
                // update train and test sets
                for (int j=0; j<lbound; j++) {
                    trainset.add(ids.get(j));
                }
                // test set
                for (int j=lbound; j<=rbound; j++) {
                    testset.add(ids.get(j));
                }
                // train set
                for (int j=rbound+1; j<ids.size(); j++) {
                    trainset.add(ids.get(j));
                }
                // compute an error
                errors[r*folds+i] = computeError(fsel);
            }
        }
        // compute an average error
        double mean = 0;
        for (int i=0; i<errors.length; i++) {
            mean += errors[i];
        }
        mean /= errors.length;
        // compute the deviation
        double dev = 0;
        for (int i=0; i<errors.length; i++) {
            dev += (errors[i]-mean)*(errors[i]-mean);
        }
        dev = Math.sqrt(dev/errors.length);
        System.out.println("Cross-validation results: mean error="+mean+" deviation="+dev);
    }

    public void estimateParameter(int maxK, int[] fsel) {
        int repeats = 100;
        int folds = 10;

        double[][] errors = new double[maxK][folds*repeats];
        for (int r=0; r<repeats; r++) {
            ArrayList<Integer> ids = new ArrayList<Integer>();
            for (int i=0; i<dataset.size(); i++) {
                ids.add(i);
            }
            Collections.shuffle(ids);

            int step = ids.size()/folds;
            for (int p=0; p<maxK; p++) {
                k = p+1;
                for (int i=0; i<folds; i++) {
                    trainset.clear();
                    testset.clear();
                    // bounds
                    int lbound = i*step;
                    int rbound = (i == folds-1) ? ids.size()-1 : (i+1)*step-1;
                    // update train and test sets
                    for (int j=0; j<lbound; j++) {
                        trainset.add(ids.get(j));
                    }
                    // test set
                    for (int j=lbound; j<=rbound; j++) {
                        testset.add(ids.get(j));
                    }
                    // train set
                    for (int j=rbound+1; j<ids.size(); j++) {
                        trainset.add(ids.get(j));
                    }
                    // compute an error
                    errors[p][r*folds+i] = computeError(fsel);
                }
            }
        }
        double[] means = new double[maxK];
        double[] devs = new double[maxK];
        for (int p=0; p<errors.length; p++) {
            // compute an average error
            double mean = 0;
            for (int i=0; i<errors[0].length; i++) {
                mean += errors[p][i];
            }
            mean /= errors[p].length;
            means[p] = mean;
            // compute the deviation
            double dev = 0;
            for (int i=0; i<errors[0].length; i++) {
                dev += (errors[p][i]-mean)*(errors[p][i]-mean);
            }
            dev = Math.sqrt(dev/errors[0].length);
            devs[p] = dev;
//			System.out.println("Cross-validation results: mean error="+mean+" deviation="+dev);
        }
        // find min average error
        double minerr = Double.POSITIVE_INFINITY;
        int minerrK = -1;
        for (int p=0; p<means.length; p++) {
            if (minerr>means[p]) {
                minerr = means[p];
                minerrK = p;
            }
        }
        System.out.println("MIN ERR: k="+(minerrK+1)+" ("+means[minerrK]+", "+devs[minerrK]+")");
        // find min deviation
        double mindev = Double.POSITIVE_INFINITY;
        int mindevK = -1;
        for (int p=0; p<means.length; p++) {
            if (mindev>means[p]) {
                mindev = means[p];
                mindevK = p;
            }
        }
        System.out.println("MIN DEV: k="+(mindevK+1)+" ("+means[mindevK]+", "+devs[mindevK]+")");
    }

    public void estimateParameterAndError(int maxK, int[] fsel) {
        int repeats = 1000;
        int folds = 10;

        double[][] errorMap = new double[repeats][2];

        ArrayList<Integer> ids = new ArrayList<Integer>();
        for (int i=0; i<dataset.size(); i++) {
            ids.add(i);
        }

        for (int r=0; r<repeats; r++) {
            Collections.shuffle(ids);
            int step = ids.size()/folds;
            int rlimit = step*(folds-1);
            // CV
            double minErr = Double.POSITIVE_INFINITY;
            int bestK = -1;
            for (int p=0; p<maxK; p++) {
                k = p+1;
                double meanErr = 0;
                for (int i=0; i<folds-1; i++) {
                    trainset.clear();
                    testset.clear();
                    // bounds
                    int lbound = i*step;
                    int rbound = (i == folds-2) ? rlimit-1 : (i+1)*step-1;
                    // update train and test sets
                    for (int j=0; j<lbound; j++) {
                        trainset.add(ids.get(j));
                    }
                    // test set
                    for (int j=lbound; j<=rbound; j++) {
                        testset.add(ids.get(j));
                    }
                    // train set
                    for (int j=rbound+1; j<rlimit; j++) {
                        trainset.add(ids.get(j));
                    }
                    // compute an error/prediction
                    meanErr += computePrediction(fsel);
                }
                meanErr /= folds-1;
                // find the best model
                if (minErr>meanErr) {
                    minErr = meanErr;
                    bestK = p;
                }
            }
            // init the hold-out set
            holdout.clear();
            for (int i=rlimit; i<ids.size(); i++) {
                holdout.add(ids.get(i));
            }
            // update the train set
            trainset.clear();
            for (int i=0; i<rlimit; i++) {
                trainset.add(ids.get(i));
            }
            // test the best model on the hold-out set - error/prediction
            errorMap[r][0] = bestK;
            errorMap[r][1] = holdoutPrediction(fsel);
        }
        // find the best generalization
        int[] hist = new int[maxK];
        for (int r=0; r<errorMap.length; r++) {
            int p = (int)errorMap[r][0];
            hist[p]++;
        }
        // the best model
        int bestK = -1;
        int maxFreq = 0;
        for (int i=0; i<hist.length; i++) {
            if (hist[i]>maxFreq) {
                maxFreq = hist[i];
                bestK = i;
            }
        }
        // get the mean error
        double mean = 0;
        for (int r=0; r<errorMap.length; r++) {
            if(errorMap[r][0]==bestK) {
                mean += errorMap[r][1];
            }
        }
        mean /= hist[bestK];
        // get the deviation
        double dev = 0;
        for (int r=0; r<errorMap.length; r++) {
            if(errorMap[r][0]==bestK) {
                dev += (errorMap[r][1]-mean)*(errorMap[r][1]-mean);
            }
        }
        dev = Math.sqrt(dev/hist[bestK]);
        System.out.println("best k="+(bestK+1)+" ( "+mean+" , "+dev+" ) ");
    }

}