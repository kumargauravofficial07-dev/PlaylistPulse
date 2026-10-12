function About() {
  return (
    <div className="page about-page">
      <div className="about-header">
        <div>
          <div className="section-label">PROJECT OVERVIEW</div>
          <h1>PlaylistPulse</h1>
          <p>
            Spotify Track Intelligence System powered by Machine Learning
          </p>
        </div>

        <div className="project-badge">
          <span>●</span>
          ML PROJECT
        </div>
      </div>

      <div className="about-grid">
        <div className="about-card overview-card">
          <div className="card-icon">P</div>
          <h2>About the Project</h2>

          <p>
            PlaylistPulse is an end-to-end Machine Learning system designed
            to analyze Spotify tracks and discover patterns in their audio
            characteristics.
          </p>

          <p>
            The system uses supervised and unsupervised machine learning
            techniques to study track popularity, classify popular tracks,
            compare different models and group songs according to their
            audio characteristics.
          </p>
        </div>

        <div className="about-card">
          <div className="card-title">
            <span className="title-icon">▣</span>
            Dataset
          </div>

          <div className="dataset-number">114,000</div>
          <div className="dataset-label">Spotify Tracks</div>

          <div className="dataset-info">
            <div>
              <strong>21</strong>
              <span>Original Columns</span>
            </div>

            <div>
              <strong>10</strong>
              <span>Audio Features</span>
            </div>

            <div>
              <strong>0</strong>
              <span>Duplicate Rows</span>
            </div>
          </div>
        </div>
      </div>

      <div className="about-card lifecycle-card">
        <div className="section-label">MACHINE LEARNING LIFECYCLE</div>
        <h2>From Data to Prediction</h2>

        <div className="lifecycle">
          <div className="lifecycle-item">
            <div className="step-number">01</div>
            <h3>Raw Data</h3>
            <p>Spotify track dataset</p>
          </div>

          <div className="lifecycle-line"></div>

          <div className="lifecycle-item">
            <div className="step-number">02</div>
            <h3>Preprocessing</h3>
            <p>Cleaning & feature preparation</p>
          </div>

          <div className="lifecycle-line"></div>

          <div className="lifecycle-item">
            <div className="step-number">03</div>
            <h3>Training</h3>
            <p>Multiple ML algorithms</p>
          </div>

          <div className="lifecycle-line"></div>

          <div className="lifecycle-item">
            <div className="step-number">04</div>
            <h3>Evaluation</h3>
            <p>Metrics & model comparison</p>
          </div>

          <div className="lifecycle-line"></div>

          <div className="lifecycle-item">
            <div className="step-number">05</div>
            <h3>Prediction</h3>
            <p>FastAPI prediction service</p>
          </div>
        </div>
      </div>

      <div className="section-label models-label">MACHINE LEARNING MODELS</div>

      <div className="models-grid">
        <div className="model-card">
          <span className="model-type">REGRESSION</span>
          <h3>Linear Regression</h3>
          <p>Predicts continuous track popularity.</p>
        </div>

        <div className="model-card">
          <span className="model-type">CLASSIFICATION</span>
          <h3>Logistic Regression</h3>
          <p>Classifies tracks into popularity categories.</p>
        </div>

        <div className="model-card">
          <span className="model-type">TREE MODEL</span>
          <h3>Random Forest</h3>
          <p>Combines multiple decision trees for classification.</p>
        </div>

        <div className="model-card">
          <span className="model-type">BOOSTING</span>
          <h3>XGBoost</h3>
          <p>Gradient boosting model for powerful predictions.</p>
        </div>

        <div className="model-card">
          <span className="model-type">UNSUPERVISED</span>
          <h3>K-Means</h3>
          <p>Groups tracks according to audio characteristics.</p>
        </div>
      </div>

      <div className="about-grid technology-grid">
        <div className="about-card">
          <div className="section-label">TECHNOLOGY STACK</div>
          <h2>Built With</h2>

          <div className="tech-list">
            <span>Python</span>
            <span>Scikit-learn</span>
            <span>Pandas</span>
            <span>NumPy</span>
            <span>FastAPI</span>
            <span>React</span>
            <span>Vite</span>
            <span>Recharts</span>
          </div>
        </div>

        <div className="about-card team-card">
          <div className="section-label">PROJECT</div>
          <h2>PlaylistPulse Team</h2>

          <div className="team-member">
            <div className="member-avatar">KG</div>
            <div>
              <strong>Kumar Gaurav</strong>
              <span>Machine Learning Project</span>
            </div>
          </div>
        </div>
      </div>

      <div className="about-footer">
        <strong>PlaylistPulse</strong>
        <span>Spotify Track Intelligence System</span>
      </div>
    </div>
  );
}

export default About;