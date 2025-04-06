# Expert Consultations Feature

This document describes the expert consultations feature added to the Statistical Model Suggester application.

## Overview

The expert consultations feature allows users to connect with statistical experts for personalized advice on their statistical analyses. This enhances the application by providing human expertise alongside algorithmic recommendations.

## Key Components

1. **Expert Profiles**: Users with statistical expertise can apply to become approved experts. 
   - Expert profiles include credentials, institution, areas of expertise, and bio.
   - Each expert must be approved by an admin before they can provide consultations.

2. **Consultation Requests**: Regular users can submit consultation requests to experts.
   - Users can link their previous analyses to provide context.
   - Users can request a specific expert or leave it open to any available expert.

3. **Consultation Management**: Experts can view assigned consultations and provide responses.
   - Consultations can be made public or kept private.
   - Public consultations appear on expert profiles, creating a knowledge base.

4. **Admin Controls**: Administrators have tools to manage the consultation ecosystem.
   - Approve/reject expert applications
   - Assign unassigned consultations to appropriate experts
   - Monitor consultation quality

## Database Changes

The implementation required several database changes:

1. Added to User model:
   - `role` field (user, expert, admin)
   - `is_approved_expert` boolean field
   - `expertise` text field
   - `bio` text field
   - `institution` text field

2. Created Consultation model:
   - Links requester and expert users
   - Optional link to a specific analysis
   - Tracks question, response, status, and public visibility

## Routes and Templates

New routes and templates were added for:
- Expert listing page
- Expert profile viewing
- Expert application
- Consultation requesting
- Consultation viewing and management
- Admin approval interface

## Usage Flow

1. **For Users Seeking Advice**:
   - Browse the experts list to find an appropriate expert
   - Create a consultation request with a specific question
   - Optionally link to a previous analysis
   - Receive notification when the expert responds

2. **For Experts**:
   - Apply to become an expert by providing credentials
   - Once approved, view assigned consultations
   - Respond to consultations with detailed advice
   - Choose to make responses public to help other users

3. **For Admins**:
   - Review expert applications and approve qualified experts
   - Assign unassigned consultations to appropriate experts
   - Monitor and manage the consultation system

## Getting Started

To use this feature after upgrading:

1. Run the database migration script:
   ```
   python scripts/migrate_db.py
   ```

2. Launch the application with the updated app_new.py:
   ```
   python app_new.py
   ```

3. Login as an admin to approve expert applications, or login as a user to browse experts and request consultations.

## Technical Implementation

The feature uses Flask blueprints to organize routes:
- `expert_routes.py`: Routes for expert-related functionality
- `admin_routes.py`: Admin controls for experts and consultations
- `user_routes.py`: User profile management
- `auth_routes.py`: Authentication functionality
- `main_routes.py`: Core application features

All database models are defined in `models.py` using SQLAlchemy. 