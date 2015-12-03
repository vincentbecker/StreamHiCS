package fullsystem;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import moa.cluster.Clustering;
import streams.UncorrelatedStream;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.DenseInstance;
import weka.core.Instance;

public class RBFDriftTest {

	//private RandomRBFGeneratorDrift stream;
	private UncorrelatedStream stream;
	private int numberSamples = 0;
	private final int numInstances = 10000;
	private final int numberOfDimensions = 2;
	private final int m = 50;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private StreamHiCS streamHiCS;
	private CorrelatedSubspacesChangeDetector cscd;
	private ClusTree ct2;

	@Before
	public void setUp() throws Exception {
		//stream = new RandomRBFGeneratorDrift();
		//stream.speedChangeOption.setValue(1);
		//stream.prepareForUse();

		stream = new UncorrelatedStream();
		stream.dimensionsOption.setValue(2);
		stream.prepareForUse();
		
		alpha = 0.15;
		epsilon = 0.15;
		threshold = 0.2;
		cutoff = 8;
		pruningDifference = 0.15;

		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(1000);
		mcs.resetLearning();
		
		ct2 = new ClusTree();
		ct2.resetLearning();

		Contrast contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, null);
		changeChecker.setCallback(streamHiCS);

		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions);
		cscd.prepareForUse();
		streamHiCS.setCallback(cscd);
	}

	@Test
	public void test() {
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			if (numberSamples % 1000 == 0) {
				System.out.println("Time: " + numberSamples);
				System.out.println("Number of microclusters: " + streamHiCS.getNumberOfElements());
				Clustering mcsR = ct2.getMicroClusteringResult();
				int s = 0;
				if(mcsR != null){
					s = mcsR.size();
				}
				System.out.println("ct2 elements: " + s);
			}
			Instance inst = stream.nextInstance();
			cscd.trainOnInstance(inst);
			/*
			DenseInstance newInst = new DenseInstance(numberOfDimensions);
			for(int i = 0; i < numberOfDimensions; i++){
				newInst.setValue(i, inst.value(i));
			}
			*/
			//ct2.trainOnInstance(inst);
			/*
			if (cscd.isWarningDetected()) {
				System.out.println("WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				System.out.println("CHANGE at " + numberSamples);
			}
			*/
			numberSamples++;
		}

		fail("Not yet implemented");
	}

}
