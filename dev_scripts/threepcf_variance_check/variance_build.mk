# Add after the project's MakefileCosmolike in a private build directory.
# The explicit objects resolve core variance and baryon symbols inside this
# extension. Python's ordinary symbols still use the platform pybind rules.
OBJECTS += $(BUILD)/halo_profile.o
OBJECTS += $(BUILD)/cosmo3D.o $(BUILD)/baryons.o
LDLIBS += -lhdf5

$(BUILD)/halo_profile.o: $(CORE)/cosmolike/halo_profile.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/cosmo3D.o: $(CORE)/cosmolike/cosmo3D.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/baryons.o: $(CORE)/cosmolike/baryons.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(TARGET): $(BUILD)/halo_profile.o
$(TARGET): $(BUILD)/cosmo3D.o $(BUILD)/baryons.o
