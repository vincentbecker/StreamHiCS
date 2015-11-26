package streamtests;

import static org.junit.Assert.*;

import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.FullSpaceContrastChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import contrast.SlidingWindowContrast;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.ArffFileStream;
import subspace.Subspace;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

/**
 * 
 * 
 * @author Vincent
 *
 */
public class RealWorldDatasets {
	private String path;
	private ArffFileStream stream;
	private int numberSamples = 0;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm() at " + numberSamples);
		}
	};
	private StreamHiCS streamHiCS;

	/*
	@Test
	public void covertype() {
		// The change points in the data are: 211840, 495141, 530895, 533642,
		// 543135, 560502
		path = "Tests/RealWorldData/Covertype_sorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 54;
		int m = 20;
		double alpha = 0.25;
		double epsilon = 0.1;
		double threshold = 0.65;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 6000;
		int checkCount = 10000;
		
		System.out.println("Covertype");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	*/
	/*
	@Test
	public void electricityNW() {
		path = "Tests/RealWorldData/ElectricityNorthWest.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 19;
		int m = 20;
		double alpha = 0.15;
		double epsilon = 0.05;
		double threshold = 0.5;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 2000;
		int checkCount = 1000;

		System.out.println("Electricity North West");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	*/
	/*
	@Test
	public void intrusionDetection10Percent() {
		// The change points in the data are: 2203, 2233, 2241, 2294, 2306,
		// 3553, 3574, 3583, 3590, 110791, 111042, 208300, 208307, 208571,
		// 209611, 209621, 211210, 492000, 492001, 492981, 494001
		// Normal is in the range of 111042 to 208299
		// Smurf is in the range of 211210 to 491999
		path = "Tests/RealWorldData/kddcup99_10_percent_sorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 34;
		int m = 20;
		double alpha = 0.15;
		double epsilon = 0;
		double threshold = 0.4;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 10000;
		int checkCount = 10000;

		System.out.println("Intrusion Detection 10%");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
		
		}
		*/
		/*
		public void intrusionDetectionFull() {
		// The change points in the data are: 
		// Normal is in the range of 
		// Smurf is in the range of 
		path = "Tests/RealWorldData/kddcup99_sorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 34;
		int m = 20;
		double alpha = 0.2;
		double epsilon = 0;
		double threshold = 0.5;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int checkCount = 50000;

		System.out.println("Intrusion Detection full");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	*/
	/*
	@Test
	public void electricityNSWSorted() {
		// The change point in the data is: 26075
		path = "Tests/RealWorldData/elecNormNew_sorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 8;
		int m = 50;
		double alpha = 0.15;
		double epsilon = 0;
		double threshold = 0.5;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int checkCount = 1000;

		System.out.println("Electricity New South Wales sorted");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	*/
	
	/*
	@Test
	public void electricityNWSUnsorted() {
		path = "Tests/RealWorldData/elecNOrmNew_unsorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 8;
		int m = 50;
		double alpha = 0.15;
		double epsilon = 0.1;
		double threshold = 0.45;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int checkCount = 1000;

		System.out.println("Electricity New South Wales unsorted");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	*/

	@Test
	public void dax30() {
		path = "Tests/RealWorldData/dax30.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		int numberOfDimensions = 30;
		int m = 50;
		double alpha = 0.15;
		double epsilon = 0.1;
		double threshold = 0.45;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 1000;
		int checkCount = 1000;

		System.out.println("DAX 30");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount);
		System.out.println();
		
		fail("Not yet implemented");
	}
	
	private void carryOutTest(int numberOfDimensions, int m, double alpha, double epsilon, double threshold, int cutoff,
			double pruningDifference, int horizon, int checkCount) {
		
		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(horizon);
		mcs.resetLearningImpl();
		Contrast contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);
		
		
		//Contrast contrastEvaluator = new SlidingWindowContrast(numberOfDimensions, m, alpha, 10000);
		
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);

		// SubspaceBuilder subspaceBuilder = new
		// FastBuilder(covarianceMatrix.length, threshold, pruningDifference,
		// contrastEvaluator);

		ChangeChecker changeChecker = new TimeCountChecker(checkCount);
		//ChangeChecker changeChecker = new FullSpaceContrastChecker(checkCount, numberOfDimensions, contrastEvaluator, 0.2, 0.1);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, callback);
		changeChecker.setCallback(streamHiCS);

		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
			if (numberSamples % checkCount == 0) {
				System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());
				System.out.println("Correlated: " + streamHiCS.toString());
				if (!streamHiCS.getCurrentlyCorrelatedSubspaces().isEmpty()) {
					for (Subspace s : streamHiCS.getCurrentlyCorrelatedSubspaces().getSubspaces()) {
						System.out.print(s.getContrast() + ", ");
					}
					System.out.println();
				}
			}
		}
	}

}
