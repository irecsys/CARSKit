package carskit.alg.cars.transformation.hybridfiltering;

import librec.data.DenseVector;

/**
 * Created by yzheng8 on 4/28/2016.
 */
public class Particle_CFPSO {
    public DenseVector pos;
    public double fitness; // minimize prediction err
    public DenseVector volocity;

    public DenseVector pos_best; // the best historical particle position
    public double fitness_best; // the best historical fitness

    public Particle_CFPSO(int len){
        pos=new DenseVector(len);
        fitness=Double.MAX_VALUE;
        volocity=new DenseVector(len);

        pos_best=new DenseVector(len);
        fitness_best=Double.MAX_VALUE;

        for(int i=0;i<len;++i){
            if(Math.random()>=0.5)
                pos.set(i, 1);
            else
                pos.set(i,0);
            pos_best.set(i,0);
        }

        volocity.init();
    }

}
