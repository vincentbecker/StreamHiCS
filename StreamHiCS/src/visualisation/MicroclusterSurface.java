package visualisation;

import java.awt.AWTException;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;

import javax.imageio.ImageIO;
import javax.swing.JPanel;
import javax.swing.Timer;
import org.apache.commons.math3.util.MathArrays;
import fullsystem.Contrast;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import statisticaltests.KolmogorovSmirnov;
import statisticaltests.StatisticalTest;
import streamdatastructures.DataBundle;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.Selection;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import streams.UncorrelatedStream;
import weka.core.DenseInstance;
import weka.core.Instance;

class MicroclusterSurface extends JPanel implements ActionListener {

	/**
	 * Default serial version id.
	 */
	private static final long serialVersionUID = 1L;
	private final int DELAY = 10;
	private Timer timer;
	private SummarisationAdapter adapter;
	private Contrast contrast;
	private StatisticalTest statTest;
	private int count = 0;
	private int imageCount = 0;
	private boolean captureScreen = false;
	private double xRange = 10;
	private double yRange = 10;
	private int[] shuffledDimensions = { 0, 1 };
	private double[][] covarianceMatrix1 = { { 1, 0.9 }, { 0.9, 1 } };
	private double[][] covarianceMatrix2 = { { 1, 0 }, { 0, 1 } };
	// private GaussianStream stream;
	private ConceptDriftStream conceptDriftStream;
	private boolean drawSlice = false;
	private DecimalFormat df = new DecimalFormat("#.##");

	public MicroclusterSurface() {

		UncorrelatedStream s1 = new UncorrelatedStream();
		s1.dimensionsOption.setValue(2);
		s1.scaleOption.setValue(10);
		s1.prepareForUse();
		GaussianStream s2 = new GaussianStream(null, covarianceMatrix1, 1);
		GaussianStream s3 = new GaussianStream(null, covarianceMatrix2, 1);

		ConceptDriftStream cds1 = new ConceptDriftStream();
		cds1.streamOption.setCurrentObject(s1);
		cds1.driftstreamOption.setCurrentObject(s2);
		cds1.positionOption.setValue(1000);
		cds1.widthOption.setValue(500);
		cds1.prepareForUse();

		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(cds1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s3);
		conceptDriftStream.positionOption.setValue(3000);
		conceptDriftStream.widthOption.setValue(500);
		conceptDriftStream.prepareForUse();

		// stream = new GaussianStream(covarianceMatrix);

		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();

		adapter = new MicroclusteringAdapter(mcs);
		this.contrast = new Contrast(20, 0.2, adapter);
		this.statTest = new KolmogorovSmirnov();
		initTimer();
	}

	private void initTimer() {

		timer = new Timer(DELAY, this);
		timer.start();
	}

	public Timer getTimer() {
		return timer;
	}

	private void doDrawing(Graphics g) {

		Graphics2D g2d = (Graphics2D) g;

		int xPixel = 0;
		int yPixel = 0;
		double[] vector;
		int w = getWidth();
		int h = getHeight();

		// Add a new instance
		vector = createAndAddInstance();
		xPixel = (int) (w * (vector[0] / xRange));
		yPixel = (int) (h * (vector[1] / yRange));
		g2d.setColor(Color.GREEN);
		g2d.fillOval(xPixel, yPixel, 10, 10);

		g2d.setColor(Color.BLUE);

		// Draw each centroid
		int weight = 0;
		Clustering microclusters = ((MicroclusteringAdapter) adapter).getMicroclusters();
		Cluster c;
		drawSlice = (count % 500 == 0);
		Selection s = null;
		if (drawSlice) {
			System.out.println(count);
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			s = adapter.getSliceIndexes(shuffledDimensions, 0.2);
			// Calculate contrast
			DataBundle projectedData = adapter.getProjectedData(shuffledDimensions[1]);
			double[] slice = adapter.getSelectedData(shuffledDimensions[1], s);
			double[] sliceWeights = adapter.getSelectedWeights(s);
			DataBundle sliceData = new DataBundle(slice, sliceWeights);
			double contrast = statTest.calculateWeightedDeviation(projectedData, sliceData);
			System.out.println(contrast);
			g2d.setColor(Color.RED);
			g2d.setFont(new Font("Calibri", Font.ITALIC, 28));
			g2d.drawString("Contrast: " + df.format(contrast), 10, h - 10);
			g2d.setColor(Color.BLUE);
		}
		for (int i = 0; i < microclusters.size(); i++) {
			c = microclusters.get(i);
			if (drawSlice && s.contains(i)) {
				g2d.setColor(Color.RED);
			}
			vector = c.getCenter();
			xPixel = (int) (w * (vector[0] / xRange));
			yPixel = (int) (h * (vector[1] / yRange));
			weight = Math.max((int) (c.getWeight() * 5), 4);
			g2d.drawOval(xPixel, yPixel, weight, weight);
			// g2d.drawString(c.getId() + "", xPixel, yPixel);
			if (drawSlice) {
				g2d.setColor(Color.BLUE);
			}
			// System.out.println("x: " + xPixel + " y: " + yPixel);
		}
	}

	@Override
	public void paintComponent(Graphics g) {

		super.paintComponent(g);
		doDrawing(g);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		revalidate();
		repaint();
		if (captureScreen) {
			if (drawSlice) {
				for (int i = 0; i < 100; i++) {
					captureApplicationImage();
				}
			} else {
				captureApplicationImage();
			}
		} else {
			if (drawSlice) {

				try {
					Thread.sleep(5000);
				} catch (InterruptedException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		}
	}

	private double[] createAndAddInstance() {
		// Instance inst = stream.nextInstance();
		Instance inst = conceptDriftStream.nextInstance();
		double x = inst.value(0);
		double y = inst.value(1);
		double offset = 5;

		/*
		 * if (count >= conceptChange) { offset = 5; } else { if (count % 2 ==
		 * 0) { offset = 3; } else { offset = 7; } }
		 */

		x += offset;
		y += offset;
		addInstance(x, y);
		count++;

		double[] point = new double[2];
		point[0] = x;
		point[1] = y;

		return point;
	}

	private void addInstance(double x, double y) {
		Instance inst = new DenseInstance(2);
		inst.setValue(0, x);
		inst.setValue(1, y);
		contrast.add(inst);
	}

	private void captureApplicationImage() {
		Rectangle screenRect = new Rectangle(this.getLocationOnScreen());
		screenRect.width = getWidth();
		screenRect.height = getHeight();
		BufferedImage capture = null;
		try {
			capture = new Robot().createScreenCapture(screenRect);
		} catch (AWTException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		try {
			ImageIO.write(capture, "jpeg", new File("C:/Users/Vincent/Desktop/Video/image" + imageCount + ".jpeg"));
		} catch (IOException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}
		imageCount++;
	}
}
