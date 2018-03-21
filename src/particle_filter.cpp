#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position
  // (based on estimates of x, y, theta and their uncertainties from GPS) and
  // all weights to 1. Add random Gaussian noise to each particle.

  default_random_engine gen;

  num_particles = 100;

  // Set standard deviations for x, y, and theta.
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Create a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    double sample_x, sample_y, sample_theta;
    Particle particle;

    particle.id = i;
    // Sample from normal distrubtions for particle attributes
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    // Set the initial weight
    particle.weight = 1.;

    particles.push_back(particle);
    //    cout << "Sample " << particle.id << " " << particle.x << " " <<
    //    particle.y
    //         << " " << particle.theta << "" << particle.weight << endl;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std[], double velocity,
                                double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: find std::normal_distribution and std::default_random_engine useful.
  // http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  // http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;

  // Set standard deviations for x, y, and theta.
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // This line creates a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (int i = 0; i < num_particles; i++) {
    Particle& particle = particles[i];
    double x = particle.x;
    double y = particle.y;
    double theta = particle.theta;

    if (abs(yaw_rate) < 0.01) {
      x += velocity * delta_t * cos(theta);
      y += velocity * delta_t * sin(theta);
      // theta is not changed
    } else {
      x +=
          velocity / yaw_rate * (+sin(theta + yaw_rate * delta_t) - sin(theta));
      y +=
          velocity / yaw_rate * (-cos(theta + yaw_rate * delta_t) + cos(theta));
      theta += yaw_rate * delta_t;
    }

    particle.x = x + dist_x(gen);
    particle.y = y + dist_y(gen);
    particle.theta = theta + dist_theta(gen);

    //    cout << i << ": " << particle.id << "" << particle.x << " " <<
    //    particle.y
    //         << " " << particle.theta << endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian
  // distribution. Read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution NOTE: The
  // observations are given in the VEHICLE'S coordinate system. The particles
  // are located according to the MAP'S coordinate system. Transformation
  // between the two systems is needed. This transformation requires both
  // rotation AND translation (but no scaling). The following is a good resource
  // for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement
  // (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html

  weights = {};
  for (unsigned int i = 0; i < num_particles; i++) {
    Particle& particle = particles[i];
    particle.associations = {};
    particle.sense_x = {};
    particle.sense_y = {};
    particle.weight = 1.;
    for (unsigned int j = 0; j < observations.size(); j++) {
      // Step 1. Transformation of particle's sensor landmark observations
      // from particle's coordinate system to map's coordinate system
      LandmarkObs observation = observations[j];

      double x = observation.x * cos(particle.theta) -
                 observation.y * sin(particle.theta) + particle.x;
      double y = observation.x * sin(particle.theta) +
                 observation.y * cos(particle.theta) + particle.y;

      particle.sense_x.push_back(x);
      particle.sense_y.push_back(y);

      // Step 2. Association of the transformed observations with the nearest
      // landmark on the map take only those landmarks in consideration which
      // are within the sensor range
      double minimum_distance = 1000.;
      Map::single_landmark_s closest_landmark;
      for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
        Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
        // the landmark is not observable by the particle
        // when the distance between the particle and the landmark is greater
        // than the sensor range the landmark can be skipped
        double distance_particle_to_landmark =
            dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
        if (distance_particle_to_landmark > sensor_range) {
          continue;
        }
        double distance_obs_to_landmark =
            dist(x, y, landmark.x_f, landmark.y_f);
        if (distance_obs_to_landmark < minimum_distance) {
          closest_landmark = landmark;
          minimum_distance = distance_obs_to_landmark;
        }
      }
      particle.associations.push_back(closest_landmark.id_i);
      // Step 3. Update of the particle weight by applying the multivariant PDF
      // for each measurement
      double obs_weight =
          gaussian2D(x, y, closest_landmark.x_f, closest_landmark.y_f,
                     std_landmark[0], std_landmark[1]);
      particle.weight *= obs_weight;
    }
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their
  // weight. NOTE: find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine generator;

  discrete_distribution<int> distribution(weights.begin(), weights.end());

  std::vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    int particle_id = distribution(generator);
    new_particles.push_back(particles[particle_id]);
    new_particles.back().id = i;
  }

  particles.swap(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
