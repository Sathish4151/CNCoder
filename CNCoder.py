import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tempfile
import base64

# Function to collect dimensions and operation type from the user
def get_input():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 3em; margin-bottom: 0;">CNCoder</h1>
            <h2 style="font-size: 1.5em; margin-top: 0;">CNC Code Generator with Simulation and AI Optimization</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("Enter the dimensions and select the operation type: ")

    operation_type = st.selectbox("Select Operation Type", ["Milling", "Drilling", "Circular Pocket", "Rectangular Pocket", "Engraving"])
    tool = st.selectbox("Select Tool", ["End Mill", "Drill", "Boring Bar", "Facing Tool", "Turning Tool"])
    material = st.selectbox("Select Material", ["Aluminum", "Steel", "Plastic", "Wood"])

    dimensions = {}
    dimensions['length'] = st.number_input("Length (mm)", min_value=0.0, step=0.1)
    dimensions['width'] = st.number_input("Width (mm)", min_value=0.0, step=0.1)
    dimensions['depth'] = st.number_input("Depth (mm)", min_value=0.0, step=0.1)
    dimensions['diameter'] = st.number_input("Diameter (mm)", min_value=0.0, step=0.1) if operation_type == "Circular Pocket" else 0.0

    return operation_type, tool, material, dimensions

# Function to generate G-code without comments
def generate_gcode(operation_type, tool, dimensions, feed_rate):
    length = dimensions['length']
    width = dimensions['width']
    depth = dimensions['depth']
    diameter = dimensions['diameter']

    gcode = [
        "G90",  # Set absolute positioning mode
        "G21",  # Set units to millimeters
        f"M06 {tool}",  # Select tool
        "M03 S1000",  # Start spindle at 1000 RPM clockwise rotation
        "G00 X10 Y10",  # Rapid move to position (10, 10)
        f"G01 Z-{depth} F{feed_rate}",  # Linear feed move to depth
        "G00 Z5",  # Rapid retract to Z position 5mm above the workpiece
        "G00 X0 Y0",  # Rapid move to home position (0, 0)
        "M05",  # Stop spindle
        "M30",  # Program end and rewind
    ]

    return gcode

# Function to generate G-code with comments
def generate_gcode_with_comments(operation_type, tool, dimensions, feed_rate):
    length = dimensions['length']
    width = dimensions['width']
    depth = dimensions['depth']
    diameter = dimensions['diameter']

    gcode = [
        "G90 ; Set absolute positioning mode",
        "G21 ; Set units to millimeters",
        f"M06 {tool} ; Select tool {tool} (tool change)",
        "M03 S1000 ; Start spindle at 1000 RPM clockwise rotation",
        "G00 X10 Y10 ; Rapid move to position (10, 10)",
        f"G01 Z-{depth} F{feed_rate} ; Linear feed move to depth -{depth}mm at feed rate {feed_rate} mm/min",
        "G00 Z5 ; Rapid retract to Z position 5mm above the workpiece",
        "G00 X0 Y0 ; Rapid move to home position (0, 0)",
        "M05 ; Stop spindle",
        "M30 ; Program end and rewind",
    ]

    return gcode

# Function to create a download link for the G-code file
def create_download_link(gcode, filename):
    gcode_text = '\n'.join(gcode)
    b64 = base64.b64encode(gcode_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}.txt">Download G-code</a>'
    return href

# Function to estimate machining time and cost
def estimate_time_and_cost(dimensions, feed_rate, material):
    volume = dimensions['length'] * dimensions['width'] * dimensions['depth']
    time = volume / feed_rate
    cost_per_minute = {"Aluminum": 0.5, "Steel": 0.75, "Plastic": 0.3, "Wood": 0.2}
    cost_usd = time * cost_per_minute.get(material, 0.5)
    exchange_rate = 82  # Example exchange rate from USD to INR
    cost_inr = cost_usd * exchange_rate
    return time, cost_inr

# Function to provide safety checks and error handling
def validate_parameters(operation_type, tool, dimensions):
    if dimensions['length'] <= 0 or dimensions['width'] <= 0 or dimensions['depth'] <= 0:
        return False, "Dimensions must be greater than zero."
    if operation_type == "Circular Pocket" and dimensions['diameter'] <= 0:
        return False, "Diameter must be greater than zero for Circular Pocket."
    return True, ""

# Main function to run the Streamlit app
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'input'

    if st.session_state.page == 'input':
        operation_type, tool, material, dimensions = get_input()

        valid, message = validate_parameters(operation_type, tool, dimensions)
        if not valid:
            st.error(message)
        elif st.button("Generate Codes"):
            feed_rate = optimize_feed_rate(dimensions, material)
            st.session_state.operation_type = operation_type
            st.session_state.tool = tool
            st.session_state.material = material
            st.session_state.dimensions = dimensions
            st.session_state.feed_rate = feed_rate
            st.session_state.gcode = generate_gcode(operation_type, tool, dimensions, feed_rate)
            st.session_state.gcode_with_comments = generate_gcode_with_comments(operation_type, tool, dimensions, feed_rate)
            st.session_state.time, st.session_state.cost = estimate_time_and_cost(dimensions, feed_rate, material)
            st.session_state.page = 'code'
            st.experimental_rerun()

    elif st.session_state.page == 'code':
        st.write(f"Optimized Feed Rate: {st.session_state.feed_rate:.2f} mm/min")
        st.write(f"Estimated Machining Time: {st.session_state.time:.2f} minutes")
        st.write(f"Estimated Cost: ₹{st.session_state.cost:.2f}")

        st.write("Generated G-code:")
        for line in st.session_state.gcode:
            st.write(line)

        # Add download button
        download_link = create_download_link(st.session_state.gcode, "generated_gcode")
        st.markdown(download_link, unsafe_allow_html=True)

        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("Run Simulation"):
                st.session_state.page = 'simulation'
                st.experimental_rerun()
        with cols[1]:
            if st.button("Explanation"):
                st.session_state.page = 'explanation'
                st.experimental_rerun()
        with cols[2]:
            if st.button("Back"):
                st.session_state.page = 'input'
                st.experimental_rerun()

    elif st.session_state.page == 'explanation':
        st.write(f"Optimized Feed Rate: {st.session_state.feed_rate:.2f} mm/min")
        st.write(f"Estimated Machining Time: {st.session_state.time:.2f} minutes")
        st.write(f"Estimated Cost: ${st.session_state.cost:.2f}")

        st.write("Generated G-code with Explanations:")
        for line in st.session_state.gcode_with_comments:
            st.write(line)

        # Add download button
        download_link = create_download_link(st.session_state.gcode, "generated_gcode")
        st.markdown(download_link, unsafe_allow_html=True)

        cols = st.columns(2)
        with cols[0]:
            if st.button("Back to Code"):
                st.session_state.page = 'code'
                st.experimental_rerun()
        with cols[1]:
            if st.button("Back to Input"):
                st.session_state.page = 'input'
                st.experimental_rerun()

    elif st.session_state.page == 'simulation':
        st.write("Simulation:")
        image_path = simulate_machining(st.session_state.gcode)
        if image_path:
            st.image(image_path)

        # Add download button
        download_link = create_download_link(st.session_state.gcode, "generated_gcode")
        st.markdown(download_link, unsafe_allow_html=True)

        # Provide a detailed summary of the process
        st.write("Summary of CNC Machining Process:")
        if st.session_state.operation_type == "Milling":
            st.write(f"1. Rapid move to position (10, 10)")
            st.write(f"2. Linear feed move to depth -{st.session_state.dimensions['depth']}mm")
            st.write(f"3. Rapid retract to Z position 5mm above the workpiece")
            st.write(f"4. Rapid move to home position (0, 0)")

        elif st.session_state.operation_type == "Drilling":
            st.write(f"1. Rapid move to position (10, 10)")
            st.write(f"2. Linear feed move to depth -{st.session_state.dimensions['depth']}mm")
            st.write(f"3. Rapid retract to Z position 5mm above the workpiece")
            st.write(f"4. Rapid move to home position (0, 0)")

        elif st.session_state.operation_type == "Circular Pocket":
            st.write(f"1. Rapid move to position (10, 10)")
            st.write(f"2. Create circular pocket with diameter {st.session_state.dimensions['diameter']}mm")
            st.write(f"3. Rapid retract to Z position 5mm above the workpiece")
            st.write(f"4. Rapid move to home position (0, 0)")

        cols = st.columns(2)
        with cols[0]:
            if st.button("Back to Code"):
                st.session_state.page = 'code'
                st.experimental_rerun()
        with cols[1]:
            if st.button("Back to Input"):
                st.session_state.page = 'input'
                st.experimental_rerun()

# Function to simulate CNC machining and generate a single image
def simulate_machining(gcode):
    # Initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # Initialize tool path lists
    x_path = []
    y_path = []
    z_path = []

    current_pos = [0, 0, 0]
    for line in gcode:
        if line.startswith("G01") or line.startswith("G00"):
            parts = line.split()
            for part in parts:
                if part.startswith("X"):
                    try:
                        current_pos[0] = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith("Y"):
                    try:
                        current_pos[1] = float(part[1:])
                    except ValueError:
                        pass
                elif part.startswith("Z"):
                    try:
                        current_pos[2] = float(part[1:])
                    except ValueError:
                        pass
            x_path.append(current_pos[0])
            y_path.append(current_pos[1])
            z_path.append(current_pos[2])

        elif line.startswith("G02"):
            parts = line.split()
            radius = float(parts[1][1:])
            u = np.linspace(0, 2 * np.pi, 100)
            x = current_pos[0] + radius * np.cos(u)
            y = current_pos[1] + radius * np.sin(u)
            z = np.full_like(x, current_pos[2])
            x_path.extend(x)
            y_path.extend(y)
            z_path.extend(z)

    # Plot the tool path
    ax.plot(x_path, y_path, z_path, label='Tool Path', color='r')
    ax.scatter(0, 0, 0, c='b', label='Start Position')  # Tool
    ax.scatter(x_path[-1], y_path[-1], z_path[-1], c='g', label='End Position')  # End Position
    ax.legend()

    # Save the plot as an image
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    image_path = temp_file.name
    temp_file.close()
    plt.savefig(image_path)

    return image_path

# Function to optimize feed rate using a simple linear regression model
def optimize_feed_rate(dimensions, material):
    # Mock data for demonstration
    X = np.array([[10, 10, 1], [20, 20, 2], [30, 30, 3], [40, 40, 4]])  # Length, Width, Depth
    y = np.array([100, 200, 300, 400])  # Corresponding feed rates

    model = LinearRegression()
    model.fit(X, y)

    length = dimensions['length']
    width = dimensions['width']
    depth = dimensions['depth']

    feed_rate = model.predict([[length, width, depth]])[0]
    material_factor = {"Aluminum": 1.0, "Steel": 0.8, "Plastic": 1.2, "Wood": 1.1}
    return feed_rate * material_factor.get(material, 1.0)

# Run the app
if __name__ == "__main__":
    main()
