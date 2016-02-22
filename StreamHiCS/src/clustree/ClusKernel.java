package clustree;

/*
 *    ClusKernel.java
 *    Copyright (C) 2010 RWTH Aachen University, Germany
 *    @author Sanchez Villaamil (moa@cs.rwth-aachen.de)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

import java.util.Arrays;
import moa.cluster.CFCluster;
import weka.core.Instance;

public class ClusKernel extends CFCluster {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public static final double EPSILON = 0.00000001;
	public static final double MIN_VARIANCE = 1e-50; // 1e-100; // 0.00000001;

	private double totalN;

	public ClusKernel(double[] point, int dim) {
		super(point, dim);
		this.totalN = 1;
	}

	protected ClusKernel(int numberDimensions) {
		super(numberDimensions);
		this.totalN = 0;
	}

	protected ClusKernel(ClusKernel other) {
		super(other);
		this.totalN = other.getTotalN();
	}

	public void add(ClusKernel other) {
		super.add(other);
		this.totalN += other.totalN;
	}

	protected void aggregate(ClusKernel other, long timeDifference, double negLambda) {
		makeOlder(timeDifference, negLambda);
		add(other);
	}

	protected void makeOlder(long timeDifference, double negLambda) {
		if (timeDifference == 0) {
			return;
		}

		double weightFactor = AuxiliaryFunctions.weight(negLambda, timeDifference);
		this.N *= weightFactor;
		for (int i = 0; i < LS.length; i++) {
			LS[i] *= weightFactor;
			SS[i] *= weightFactor;
		}
	}

	public double calcDistance(ClusKernel other) {
		// TODO: (Fernando, Felix) Adapt the distance function to the new
		// algorithmn.

		double N1 = this.getWeight();
		double N2 = other.getWeight();

		double[] thisLS = this.LS;
		double[] otherLS = other.LS;

		double res = 0.0;
		for (int i = 0; i < thisLS.length; i++) {
			double substracted = (thisLS[i] / N1) - (otherLS[i] / N2);
			res += substracted * substracted;
		}

		// TODO INFO: added sqrt to the computation [PK 10.09.10]
		return Math.sqrt(res);
	}

	private double getTotalN() {
		return totalN;
	}

	protected boolean isEmpty() {
		return this.totalN == 0;
	}

	protected void clear() {
		this.totalN = 0;
		this.N = 0.0;
		Arrays.fill(this.LS, 0.0);
		Arrays.fill(this.SS, 0.0);
	}

	protected void overwriteOldCluster(ClusKernel other) {
		this.totalN = other.totalN;
		this.N = other.N;
		AuxiliaryFunctions.overwriteDoubleArray(this.LS, other.LS);
		AuxiliaryFunctions.overwriteDoubleArray(this.SS, other.SS);

	}

	@Override
	public double getWeight() {
		return this.N;
	}

	@Override
	public CFCluster getCF() {
		return this;
	}

	public double[] getCenter() {
		assert (!this.isEmpty());
		double res[] = new double[this.LS.length];
		double weightedSize = this.getWeight();
		for (int i = 0; i < res.length; i++) {
			res[i] = this.LS[i] / weightedSize;
		}
		return res;
	}

	// @Override
	// public double getInclusionProbability(Instance instance) {
	//
	// double dist = calcNormalizedDistance(instance.toDoubleArray());
	// double res = AuxiliaryFunctions.distanceProbabilty(dist, LS.length);
	// assert (res >= 0.0 && res <= 1.0) : "Bad confidence " + res + " for"
	// + " distance " + dist;
	//
	// return res;
	// }

	@Override
	public double getInclusionProbability(Instance instance) {
		// trivial cluster
		if (N == 1) {
			double distance = 0.0;
			for (int i = 0; i < LS.length; i++) {
				double d = LS[i] - instance.value(i);
				distance += d * d;
			}
			distance = Math.sqrt(distance);
			if (distance < EPSILON)
				return 1.0;
			return 0.0;
		} else {
			double dist = calcNormalizedDistance(instance.toDoubleArray());
			if (dist <= getRadius()) {
				return 1;
			} else {
				return 0;
			}
			// double res = AuxiliaryFunctions.distanceProbabilty(dist,
			// LS.length);
			// return res;
		}
	}

	@Override
	public double getRadius() {
		// trivial cluster
		if (N == 1)
			return 0;

		return getDeviation() * radiusFactor;
	}

	private double getDeviation() {
		double[] variance = getVarianceVector();
		double sumOfDeviation = 0.0;
		for (int i = 0; i < variance.length; i++) {
			double d = Math.sqrt(variance[i]);
			sumOfDeviation += d;
		}
		return sumOfDeviation / variance.length;
	}

	private double[] getVarianceVector() {
		double[] res = new double[this.LS.length];
		for (int i = 0; i < this.LS.length; i++) {
			double ls = this.LS[i];
			double ss = this.SS[i];

			double lsDivN = ls / this.getWeight();
			double lsDivNSquared = lsDivN * lsDivN;
			double ssDivN = ss / this.getWeight();
			res[i] = ssDivN - lsDivNSquared;

			// Due to numerical errors, small negative values can occur.
			// We correct this by settings them to almost zero.
			if (res[i] <= 0.0) {
				if (res[i] > -EPSILON) {
					res[i] = MIN_VARIANCE;
				}
			} else {

			}
		}
		return res;
	}

	// ???????
	private double calcNormalizedDistance(double[] point) {
		double[] center = getCenter();
		double res = 0.0;

		for (int i = 0; i < center.length; i++) {
			double diff = center[i] - point[i];
			res += (diff * diff);// variance[i];
		}
		return Math.sqrt(res);
	}

}
