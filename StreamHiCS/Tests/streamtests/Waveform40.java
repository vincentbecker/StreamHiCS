package streamtests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.generators.WaveformGenerator;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.FastBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class Waveform40 {

	private WaveformGenerator stream;
	private StreamHiCS streamHiCS;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("onAlarm().");
		}
	};
	private int numInstances = 10000;
	private int numberSamples = 0;
	private int numberOfDimensions = 40;

	@Before
	public void setUp() throws Exception {
		stream = new WaveformGenerator();
		stream.addNoiseOption.set();
		stream.prepareForUse();

		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();

		double alpha = 0.15;
		double epsilon = 0;
		double threshold = 0.50;
		int cutoff = 8;
		double pruningDifference = 0.15;

		Contrast contrastEvaluator = new MicroclusterContrast(20, alpha, mcs);
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, pruningDifference, contrastEvaluator);
		//SubspaceBuilder subspaceBuilder = new FastBuilder(numberOfDimensions, threshold, cutoff, pruningDifference,
		//		contrastEvaluator);
		this.streamHiCS = new StreamHiCS(epsilon, threshold, contrastEvaluator, subspaceBuilder, changeChecker,
				callback);
		changeChecker.setCallback(streamHiCS);
	}

	@Test
	public void test() {
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
			if (numberSamples != 0 && numberSamples % 1000 == 0) {
				SubspaceSet correlatedSubspaces = streamHiCS.getCurrentlyCorrelatedSubspaces();
				System.out.println("Correlated: " + correlatedSubspaces.toString());
				if (!correlatedSubspaces.isEmpty()) {
					for (Subspace s : correlatedSubspaces.getSubspaces()) {
						System.out.print(s.getContrast() + ", ");
					}
					System.out.println();
				}
			}
		}

		fail("Not implemented yet. ");
	}

}
